# Inner-TRAIN probe-length sensitivity

Frozen trainedepoch2checkpoint; only16innerTRAINselection identities. Fivefull-lengthgallerywindows/person; tenprobes each truncatedto25/50/75/100storedfeature rows thenzero-padded inallfivechannels. No newthreshold orminimumlength selected.

| Probe events | Pooled discrete EER | Macro discrete EER |
|---|---:|---:|
|25|45.625%|36.4375%|
|50|30.625%|26.8542%|
|75|18.2083%|16.125%|
|100|14.9167%|11.25%|

All160genuine/2400impostor comparisons percondition are fromTRAINselection, alreadyexposed tocheckpointselection. No calibration/DEV/test arrays loaded. This audit was triggered by disclosedDEVcount failure but does not useDEVtimings/scores.

The sourcepadsto100butdoesnotmaskpadding: learnedkey0embedding, bidirectionalGRUs,attention,BN/conv/pooling respond tozero tails. Removingprobeinformationandshiftingpaddingdistribution areconfounded. Truncatedstoredresiduals differfromtrue shortenedrawsynthesis becausefallbackRNGdraworder changes. Tenseparate generatedtests verify authorbasepadding,wirevalidity,inputpreservation,andthisRNGdifference. No safe capturelength orDEVperformance claim.
