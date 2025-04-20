import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import japanize_matplotlib
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

import slack_sdk
import emoji
import MeCab
_mecab_path = "-r /dev/null -d /opt/homebrew/lib/mecab/dic/ipadic-neologd"

### how to get requests ###
# client = slack_sdk.WebClient(token={your token})
# request = client.conversations_history(channel={your target channel}, limit=999)

def print_channel_ids(client, token):
    _list = client.conversations_list(token=token)
    for _c in _list.get('channels'):
        print(f"id: {_c['id']}, name: {_c['name']}")

def get_real_name(client, token):
    # recognize all members by id
    real_names = {}
    all_members = client.users_list(token=token)['members']
    for _u in all_members:
        if (_u['name'] != 'slackbot') & _u['deleted'] == False:
            real_names[_u['id']] = _u['real_name']
        else:
            pass
    return real_names

# main Morphological Analysis tools
# joint all messages with the given range
def joint_messages_with_range(request, start=None, stop=None):
    if start is None:
        start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    if stop is None:
        stop = datetime.now(tz=ZoneInfo("Asia/Tokyo"))

    # get messages within the time range
    messages = []
    for _m in request["messages"]:
        _ts = float(_m['ts'])
        _dt = datetime.fromtimestamp(_ts, tz=ZoneInfo("Asia/Tokyo"))
        if start <= _dt <= stop:
            if 'blocks' in _m.keys():
                for _b in _m['blocks']:
                    if 'elements' in _b.keys():
                        for _e in _b['elements']:
                            if 'elements' in _e.keys():
                                for _ee in _e['elements']:
                                    if 'text' in _ee.keys():
                                        messages.append(_ee['text'])
                            else:
                                pass
            else:
                pass
    return ",".join(messages)

# tokenize text
def tokenize_with_pos(text):
    mecab = MeCab.Tagger(_mecab_path)
    nodes = mecab.parseToNode(text)
    tokens = []
    while nodes:
        if nodes.surface != "":
            tokens.append((nodes.surface, nodes.feature.split(',')[0], nodes.feature.split(',')[1]))
        nodes = nodes.next
    return tokens

def reduce_tokens(tokens_with_pos, part_of_speech, is_special_part=False):
    uninteresting_tokens = [
        'the', 'to', 'is', 'are', 'and', 'We', 'for', 'has', 'will', 'I',
        'on', 'you', 'as', 'if', 'but', 'was', 'some', 'we', 'in',
        'an', 'You', 'like', 'need', 'been', 'have', 'me', 'it',
        'that', 'by', 'This', 'any', 'with', 'now', 'not', 'or',
        'The', 'at', 'of', 'It', 'here', 'one', 'I will', 'them',
        'all', 'someone', 'take', 'just', 'other', 'know', 'be',
        'about', 'who', 'able', 'next', 'can', 'ask', 'am', 'if',
        'If', 'from', 'These', 'As', 'as', 'what', 'only', 'made',
        'write', 'would', 'below', 'a', 'this', 'which', "I've",
        'https', 'www', '.com', 'ac.jp', 'u.',
        'こと', 'よう', 'ため', 'ない', 'いる', 'する', 'ある', 'の',
        'これ', 'それ', 'あれ', 'どれ', 'もの', 'そう', 'ところ'
    ]

    df = pd.DataFrame(tokens_with_pos, columns=['token', 'pos', 'pos2'])
    if is_special_part: # proper nouns
        df = df[df['pos2'] == '固有名詞']
    else:
        df = df[df['pos'] == part_of_speech]
    df = df[~df['token'].isin(uninteresting_tokens)]
    df = df['token'].value_counts().reset_index()
    df.columns = ['token', 'count']
    df = df[df['count'] > 1]
    df = df.sort_values('count', ascending=False)
    df = df.reset_index(drop=True)
    return df

# addtional tools for emoji
def get_emoji_data(request, real_names):
    # dump all messages and reactions
    ts = []
    user = []
    ruser = []
    actions = []
    reactions = []
    for _m in request['messages']:
        ts.append(_m['ts'])
        user.append(real_names[_m['user']])
        if 'blocks' in _m:
            for _b in _m['blocks']:
                if 'elements' in _b:
                    _name = []
                    for _e in _b['elements']:
                        if 'elements' in _e:
                            #print(_e)
                            for _e2 in _e['elements']:
                                if _e2['type'] == 'emoji':
                                    _name.append(_e2['name'])
                                elif _e2['type'] == 'text':
                                    _name.append('text')
                        else:
                            actions.append('no_emoji')
                    _names = ",".join(_name)
                    actions.append(_names)
                else:
                    actions.append('no_emoji')
        else:
            actions.append('no_emoji')
        if 'reactions' in _m:
            _reac = []
            _ruser = []
            for _r in _m['reactions']:
                # print(f"{_r['name']}: {len(_r['users'])}")
                if len(_r['users']) > 1:
                    for _r2 in _r['users']:
                        _reac.append(_r['name'])
                        _ruser.append(real_names[_r2])
                else:
                    _reac.append(_r['name'])
                    _ruser.append(real_names[_r['users'][0]])
            _reacs = ",".join(_reac)
            reactions.append(_reacs)
            _rusers = ",".join(_ruser)
            ruser.append(_rusers)
        else:
            reactions.append("no_reaction")
            ruser.append("no_reply")
    ts = np.array(ts)
    user = np.array(user)
    ruser = np.array(ruser)
    actions = np.array(actions)
    reactions = np.array(reactions)

    # create DataFrame
    df = pd.DataFrame({
        'ts': ts,
        'user': user,
        'ruser': ruser,
        'actions': actions,
        'reactions': reactions
    })
    return df
