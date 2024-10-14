import torch

from dqn.Utility import *

t1 = Transition(
    torch.tensor([[1, 2, 3]]),
    torch.tensor([[1, 2]]),
    torch.tensor([[4, 5, 6]]),
    torch.tensor([[1]]),
    torch.tensor([[True]])
)

t2 = Transition(
    torch.tensor([[4, 5, 6]]),
    torch.tensor([[3, 4]]),
    torch.tensor([[7, 8, 9]]),
    torch.tensor([[2]]),
    torch.tensor([[False]])
)

def test_pack_transition_batch():
    '''
    保证 Transition 沿 Batch 维度正确合并
    '''

    t_desire = Transition(
        torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]]),
        torch.tensor([[1, 2], [3, 4], [1, 2]]),
        torch.tensor([[4, 5, 6], [7, 8, 9], [4, 5, 6]]),
        torch.tensor([[1], [2], [1]]),
        torch.tensor([[True], [False], [True]])
    )  

    t_test = pack_transition_batch([t1, t2, t1])
    
    assert t_desire == t_test

def test_make_transition():
    '''
    保证基于 numpy 数组的 Transition 能正确创建 
    '''

    t_desire1 = t1

    t_test1 = make_transition_from_numpy(
        np.array([1, 2, 3]),
        np.array([1, 2]),
        np.array([4, 5, 6]),
        1,
        True
    )

    assert t_desire1 == t_test1

    t_desire2 = pack_transition_batch([t1, t2, t1])

    t_test2 = make_transition_from_numpy(
        np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]]),
        np.array([[1, 2], [3, 4], [1, 2]]),
        np.array([[4, 5, 6], [7, 8, 9], [4, 5, 6]]),
        np.array([[1], [2], [1]]),
        np.array([[True], [False], [True]])
    )

    assert t_desire2 == t_test2

def test_reply_queue():
    '''
    保证批量随机采样得到的 Transition 维度正确为 B x O
    '''
    
    rq = ReplyQueue(3)
    rq.append(t1)
    rq.append(t2)
    rq.append(t1)

    sample = rq.sample(2)

    assert sample.state.shape == (2, 3)
    assert sample.action.shape == (2, 2)
    assert sample.next_state.shape == (2, 3)
    assert sample.reward.shape == (2, 1)
    assert sample.done.shape == (2, 1)

    pass

if __name__ == "__main__":
    test_make_transition()
    test_pack_transition_batch()
    test_reply_queue()
