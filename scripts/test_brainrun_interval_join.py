import copy
import pytest
from brainrun_interval_join import IntervalJoin, join_intervals


def row(start, stop, **kwargs):
    return {'start_ms':start,'stop_ms':stop,'user':('string','u'),
            'device':('string','d'),'task':'Mathisis',**kwargs}


def game(start,stop,name='g',**kwargs):return row(start,stop,id=('string',name),**kwargs)


def test_boundaries_and_adjacent_zero_length_ambiguity():
    results=join_intervals([game(0,10,'a'),game(10,20,'b')],
                          [row(0,10),row(10,10),row(10,20),row(9,11),row(20,20),row(21,21)])
    assert [r['status'] for r in results]==['matched','ambiguous','matched','no_game','matched','no_game']
    assert results[1]['candidate_count']==2 and results[1]['game_id'] is None
    assert results[4]['game_id']==('string','b')


def test_overlap_never_chooses_arbitrary_game():
    results=join_intervals([game(0,100,'long'),game(20,30,'short')],[row(22,25),row(40,45)])
    assert results[0]=={'status':'ambiguous','game_id':None,'candidate_count':2}
    assert results[1]['game_id']==('string','long')


@pytest.mark.parametrize('change',[{'task':'Memoria'},{'user':('string','other')},
                                 {'device':('string','other')},{'device':('int32',12)}])
def test_exact_context_keys_only(change):
    gesture=row(5,10,device=('string','12'));gesture.update(change)
    assert join_intervals([game(0,100,device=('string','12'))],[gesture])[0]['status']=='no_game'


def test_index_and_join_do_not_mutate_inputs():
    games=[game(10,20,'b'),game(0,10,'a')];gestures=[row(12,13),row(1,2)]
    before=copy.deepcopy((games,gestures))
    result=join_intervals(games,gestures)
    assert (games,gestures)==before
    assert [r['game_id'] for r in result]==[('string','b'),('string','a')]
    index=IntervalJoin(games);games[0]['stop_ms']=11
    assert index.match(gestures[0])['status']=='matched'


@pytest.mark.parametrize('start,stop',[(True,3),(0,False),(-1,2),(3,2),(1.0,2),(0,float('inf')),(0,float('nan'))])
def test_invalid_clocks_throw(start,stop):
    with pytest.raises(ValueError):IntervalJoin([game(start,stop)])
    with pytest.raises(ValueError):IntervalJoin([]).match(row(start,stop))


def test_zero_duration_game_rejected_and_duplicate_ids_global():
    with pytest.raises(ValueError):IntervalJoin([game(1,1)])
    with pytest.raises(ValueError):IntervalJoin([game(0,10),game(20,30,device=('string','other'))])


@pytest.mark.parametrize('change',[{'task':''},{'task':None},{'user':'u'},{'device':('int32',True)},
                                 {'user':('objectid','bad')}])
def test_invalid_context_rejected(change):
    with pytest.raises(ValueError):IntervalJoin([game(0,1,**change)])
    with pytest.raises(ValueError):IntervalJoin([]).match(row(0,0,**change))


def test_empty_index_explicit_unmatched():
    assert IntervalJoin([]).match(row(0,0))=={'status':'no_game','game_id':None,'candidate_count':0}
