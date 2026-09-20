"""Pure sampling and pinned author-model loading for the HMOG adaptation.

No dataset loader, checkpoint lookup, or training runs execute on import.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parents[1] / 'research/benchmarks/references/hmog'
SOURCE_PREFIX = 'behaveformer-experiments_keystroke_imu_combined_HMOGDB_imu_acc_gyr_'
MODEL_HASH = '9adfae16c8b7f28ab6cc2a68bc1b8977c1de9533f2a9fb41f889b4f45aa15404'
LOSS_HASH = 'd539b4ffc5b05a838bc65fc0b15c0c69bef91d5f045ee88d4578568795c5e873'
FIT_IDS = frozenset(('717868', '526319', '986737', '539502'))


def load_author_classes(source_dir=SOURCE_DIR):
    """Verify pinned bytes before compiling architecture and exact loss only."""
    import torch

    model_path = source_dir / (SOURCE_PREFIX + 'model.py')
    loss_path = source_dir / (SOURCE_PREFIX + 'train.py')
    model_bytes, loss_bytes = model_path.read_bytes(), loss_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != MODEL_HASH:
        raise ValueError('Author architecture checksum mismatch')
    if hashlib.sha256(loss_bytes).hexdigest() != LOSS_HASH:
        raise ValueError('Author loss source checksum mismatch')
    # Execute the verified bytes, avoiding a check/reopen race and never
    # importing the training script's top-level dataset preparation.
    spec = importlib.util.spec_from_loader('hmog_pinned_author_model', loader=None)
    model_module = importlib.util.module_from_spec(spec)
    exec(compile(model_bytes, str(model_path), 'exec'), model_module.__dict__)
    tree = ast.parse(loss_bytes)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TripletLoss']
    if len(nodes) != 1:
        raise ValueError('Expected exactly one author loss class')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(loss_path), 'exec'), namespace)
    return model_module.Model, namespace['TripletLoss']


class TripletSampler:
    """Source-style identity/session-uniform sampling using actual availability.

    Metadata corresponds one-to-one to window rows. Only fit-cohort sessions
    9–16 may enter global training, even if enrollment arrays are available.
    Accounts lacking two sessions can be negatives but cannot be anchors.
    """

    def __init__(self, subjects, sessions, roles, *, seed):
        if not (len(subjects) == len(sessions) == len(roles)) or not len(subjects):
            raise ValueError('Empty or inconsistent window metadata')
        self.groups = {}
        for index, (subject, session, role) in enumerate(zip(subjects, sessions, roles)):
            subject = str(subject)
            if (subject not in FIT_IDS or role != 'train_fit'
                    or not isinstance(session, (int, np.integer))
                    or isinstance(session, (bool, np.bool_)) or not 9 <= session <= 16):
                raise PermissionError('Global training requires fit identity TRAIN probe windows')
            self.groups.setdefault(subject, {}).setdefault(int(session), []).append(index)
        self.subjects = sorted(self.groups)
        self.anchors = [s for s in self.subjects if len(self.groups[s]) >= 2]
        if len(self.subjects) < 2 or not self.anchors:
            raise ValueError('Cross-session genuine and cross-identity negative sampling infeasible')
        self.rng = np.random.default_rng(seed)

    def sample(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('Positive integer triplet count required')
        output = []
        for _ in range(count):
            subject = str(self.rng.choice(self.anchors))
            a_session, p_session = self.rng.choice(sorted(self.groups[subject]), 2, replace=False)
            other = str(self.rng.choice([s for s in self.subjects if s != subject]))
            n_session = int(self.rng.choice(sorted(self.groups[other])))
            output.append([self.rng.choice(self.groups[subject][int(a_session)]),
                           self.rng.choice(self.groups[subject][int(p_session)]),
                           self.rng.choice(self.groups[other][n_session])])
        return np.asarray(output, dtype=np.int64)
