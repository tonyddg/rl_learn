import torch
from ..dqn.BaseDQN import *

ACTION_NUM = 2
STATE_DIM = 4

model = BaseDQN(HyperParam(state_dim = STATE_DIM, action_num = ACTION_NUM))

def assert_legal_action(output: torch.Tensor, batch_size: int):
    assert output.size() == (batch_size, 1)
    assert (output < ACTION_NUM).all() and (output >= 0).all()

def test_take_action(batch_size: int = 3):
    '''
    保证 take_action 方法输入 B x S 的张量, 输出 B x 1 的张量, 且值为合法动作
    '''
    test_random_action = model._take_random_action(torch.rand((batch_size, STATE_DIM)))
    assert_legal_action(test_random_action, batch_size)

    test_valuable_action = model._take_valuable_action(torch.rand((batch_size, STATE_DIM)))
    assert_legal_action(test_valuable_action, batch_size)

    test_action = model.take_action_vectorize(torch.rand((batch_size, STATE_DIM)))
    assert_legal_action(test_action, batch_size)

def test_get_td_target(batch_size: int = 3):
    '''
    保证 get_td_target 方法输出为 B x 1
    '''
    test_td_target = model._get_td_target(random_transition(STATE_DIM, 1, batch_size))
    assert test_td_target.size() == (batch_size, 1)

def test_get_predict(batch_size: int = 3):
    '''
    保证 get_predict 方法输出为 B x 1
    '''
    test_td_target = model._get_predict(random_transition(STATE_DIM, 1, batch_size))
    assert test_td_target.size() == (batch_size, 1)

def test_train(batch_size: int = 3):
    '''
    确定 train 方法的运行不会出错
    '''
    model._batch_update(random_transition(STATE_DIM, 1, batch_size))
