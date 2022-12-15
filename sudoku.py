#!/usr/bin/env python
# coding: utf-8

# # 数独は集合演算だよねってお話

# In[ ]:


#必要なライブラリのインポート
import functools
import itertools
import numpy
import tqdm


# In[ ]:


# ビットカウントを行う関数
def count_bits(bits:numpy.ndarray) -> numpy.ndarray:
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555)
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333)
    bits = (bits & 0x0F0F0F0F) + (bits >> 4 & 0x0F0F0F0F)
    bits = (bits & 0x00FF00FF) + (bits >> 8 & 0x00FF00FF)
    bits = (bits & 0x0000FFFF) + (bits >> 16 & 0x0000FFFF)
    return bits


# In[ ]:


# 最上位ビットを取得する関数
def calc_msb(bits:numpy.ndarray) -> numpy.ndarray:
    msb = numpy.zeros(bits.shape)
    msb += numpy.where(bits & 0xFFFF0000, 16, 0)
    msb += numpy.where(bits & 0xFF00FF00, 8, 0)
    msb += numpy.where(bits & 0xF0F0F0F0, 4, 0)
    msb += numpy.where(bits & 0xCCCCCCCC, 2, 0)
    msb += numpy.where(bits & 0xAAAAAAAA, 1, 0)
    return msb + 1


# In[ ]:


# テーブルから候補を生成する関数
def table_to_candidates(table:numpy.ndarray) -> numpy.ndarray:
    mask = table != 0
    candidates = numpy.full(table.shape, 0b111111111)
    candidates[mask] = 1 << (table[mask] - 1)
    return candidates


# In[ ]:


# 候補からテーブルを生成する関数
def candidates_to_table(candidates:numpy.ndarray) -> numpy.ndarray:
    mask = count_bits(candidates) == 1
    table = numpy.zeros(candidates.shape, dtype='int32')
    table[mask] = calc_msb(candidates[mask])
    return table


# In[ ]:


# 候補を整理する関数
def arrange_candidates(candidates:numpy.ndarray, arranged:numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:

    # 各マスに対して処理
    for row, col in zip(*numpy.where(~arranged)):

        # 候補が一つでないときの処理
        if count_bits(candidates[row,col]) != 1:

            # 自身が属している行・列・箱について候補となっている数字を計算
            set_row = functools.reduce(lambda x,y: x | y, [
                candidates[row,i] for i in range(9) if i != col
            ])
            set_col = functools.reduce(lambda x,y: x | y, [
                candidates[i,col] for i in range(9) if i != row
            ])
            set_box = functools.reduce(lambda x,y: x | y, [
                candidates[i,j]
                for i in range(3 * (row // 3), 3 * (row // 3 + 1))
                for j in range(3 * (col // 3), 3 * (col // 3 + 1))
                if i != row or j != col
            ])

            # 求めた和集合のうち、ある数字が含まれないものが存在する場合、自身のマスに入るのはその数字である
            candidate = candidates[row,col] & ((~set_row) | (~set_col) | (~set_box))
            if candidate != 0: candidates[row,col] = candidate

        # 上記を行った結果候補が一つとなったときの処理
        if count_bits(candidates[row,col]) == 1:

            # 整理済みであることを記録
            arranged[row, col] = True
        
            # マスが属する列について候補を削除
            for i in range(9):
                if i != row:
                    candidates[i,col] &= ~candidates[row,col]

            # マスが属する行について候補を削除
            for i in range(9):
                if i != col:
                    candidates[row,i] &= ~candidates[row,col]

            # マスが属する箱について候補を削除
            box_row = 3 * (row // 3)
            box_col = 3 * (col // 3)
            for i, j in itertools.product(range(3), range(3)):
                if box_row + i != row and box_col + j != col:
                    candidates[box_row+i,box_col+j] &= ~candidates[row,col]
    
    # 処理後の候補と各マスで整理済みかを返す
    return candidates, arranged


# In[ ]:


# 配列の分割を行うジェネレータ
def divided_array(array:numpy.ndarray) -> numpy.ndarray:

    # 集合を分割する場合の数を計算
    condnum = 1 << array.shape[0]

    # そのインデックスで計算を飛ばすかを記録
    skip = numpy.full(condnum, False)

    # 各インデックスについて、ビットが0か1かで集合を分割
    for i in range(condnum):
        if skip[i]: continue
        skip[i] = True
        skip[i^(condnum-1)] = True
        mask = i & (1 << numpy.arange(array.shape[0]))
        yield array[mask!=0], array[mask==0]


# In[ ]:


# テーブルの各行に対して集合分割を行う関数
def divide_set_row(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各行に対して処理
    for row in range(9):

        # 行において、候補が一つに定まっていないマスの集合を取得
        index = numpy.arange(9)
        index = index[count_bits(candidates[row,:])!=1]

        # マスの集合を分割し、それぞれの場合について処理
        for index1, index2 in divided_array(index):

            # 分割した集合の要素数が両方2以上の場合に処理
            if index1.shape[0] <= 1: continue
            if index2.shape[0] <= 1: continue

            # マスと数字の集合計算
            set1 = functools.reduce(lambda x,y: x | y, candidates[row,index1])
            if count_bits(set1) == index1.shape[0]:
                candidates[row,index2] &= ~set1
            
            # マスと数字の集合計算
            set2 = functools.reduce(lambda x,y: x | y, candidates[row,index2])
            if count_bits(set2) == index2.shape[0]:
                candidates[row,index1] &= ~set2

    # 処理後の候補を返す
    return candidates


# In[ ]:


# テーブルの各列に対して集合分割を行う関数
def divide_set_col(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各列に対して処理
    for col in range(9):

        # 列において、候補が一つに定まっていないマスの集合を取得
        index = numpy.arange(9)
        index = index[count_bits(candidates[:,col])!=1]

        # マスの集合を分割し、それぞれの場合について処理
        for index1, index2 in divided_array(index):

            # 分割した集合の要素数が両方2以上の場合に処理
            if index1.shape[0] <= 1: continue
            if index2.shape[0] <= 1: continue

            # マスと数字の集合計算
            set1 = functools.reduce(lambda x,y: x | y, candidates[index1,col])
            if count_bits(set1) == index1.shape[0]:
                candidates[index2,col] &= ~set1
            
            # マスと数字の集合計算
            set2 = functools.reduce(lambda x,y: x | y, candidates[index2,col])
            if count_bits(set2) == index2.shape[0]:
                candidates[index1,col] &= ~set2

    # 処理後の候補を返す
    return candidates


# In[ ]:


# テーブルの各箱に対して集合分割を行う関数
def divide_set_box(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各箱に対して処理
    for box_row, box_col in itertools.product(range(3), range(3)):

        # 列において、候補が一つに定まっていないマスの集合を取得
        index = numpy.arange(9)
        index = index[count_bits(numpy.ravel(candidates[3*box_row:3*(box_row+1),3*box_col:3*(box_col+1)]))!=1]

        # マスの集合を分割し、それぞれの場合について処理
        for index1, index2 in divided_array(index):

            # 分割した集合の要素数が両方2以上の場合に処理
            if index1.shape[0] <= 1: continue
            if index2.shape[0] <= 1: continue

            # インデックスを変換
            box_row1, box_col1 = numpy.unravel_index(index1, (3, 3))
            box_row1 += 3 * box_row
            box_col1 += 3 * box_col
            
            box_row2, box_col2 = numpy.unravel_index(index2, (3, 3))
            box_row2 += 3 * box_row
            box_col2 += 3 * box_col

            # マスと数字の集合計算
            set1 = functools.reduce(lambda x,y: x | y, candidates[box_row1,box_col1])
            if count_bits(set1) == index1.shape[0]:
                candidates[box_row2,box_col2] &= ~set1
            
            # マスと数字の集合計算
            set2 = functools.reduce(lambda x,y: x | y, candidates[box_row2,box_col2])
            if count_bits(set2) == index2.shape[0]:
                candidates[box_row1,box_col1] &= ~set2

    # 処理後の候補を返す
    return candidates


# In[ ]:


# テーブルの各箱行に対して集合差分を行う関数
def subtract_set_rowbox(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各箱に対して処理
    for box_row, box_col in itertools.product(range(3), range(3)):

        # 箱の内容を取得
        box = candidates[(3*box_row):(3*box_row+3),(3*box_col):(3*box_col+3)]

        # 箱の中の各行について、その行にしか含まれない候補を洗い出す
        mask = numpy.array([functools.reduce(lambda x,y: x | y, box[i,:]) for i in range(3)])
        mask = numpy.array([
            mask[0] & ~(mask[1] | mask[2]),
            mask[1] & ~(mask[2] | mask[0]),
            mask[2] & ~(mask[0] | mask[1]),
        ])

        # 先ほどの候補は、同じ箱行に属する箱において同じ行に入らない
        for i in range(3):
            if i == box_col: continue
            candidates[(3*box_row):(3*box_row+3),(3*i):(3*i+3)] &= ~mask[:,None]

        # 同じ箱行に属する自身以外の箱について、各行で候補となっている数字を洗い出す
        mask = numpy.zeros(3, dtype='int32')
        for i in range(3):
            if i == box_col: continue
            box = candidates[(3*box_row):(3*box_row+3),(3*i):(3*i+3)]
            mask |= numpy.array([functools.reduce(lambda x,y: x | y, box[i,:]) for i in range(3)])

        # 自身以外の箱の各行について、その行にのみ含まれない候補を洗い出す
        mask = numpy.array([
            (mask[1] | mask[2]) & ~mask[0],
            (mask[2] | mask[0]) & ~mask[1],
            (mask[0] | mask[1]) & ~mask[2],
        ])

        # 自身以外の箱の各行についてその行にのみ含まれない数字は、自身の箱のその行にしか含まれない
        mask = numpy.array([
            mask[1] | mask[2],
            mask[2] | mask[0],
            mask[0] | mask[1],
        ])
        candidates[(3*box_row):(3*box_row+3),(3*box_col):(3*box_col+3)] &= ~mask[:,None]
    
    # 処理後の候補を返す
    return candidates


# In[ ]:


# テーブルの各箱列に対して集合差分を行う関数
def subtract_set_colbox(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各箱に対して処理
    for box_row, box_col in itertools.product(range(3), range(3)):

        # 箱の内容を取得
        box = candidates[(3*box_row):(3*box_row+3),(3*box_col):(3*box_col+3)]

        # 箱の中の各列について、その列にしか含まれない候補を洗い出す
        mask = numpy.array([functools.reduce(lambda x,y: x | y, box[:,i]) for i in range(3)])
        mask = numpy.array([
            mask[0] & ~(mask[1] | mask[2]),
            mask[1] & ~(mask[2] | mask[0]),
            mask[2] & ~(mask[0] | mask[1]),
        ])

        # 先ほどの候補は、同じ箱列に属する箱において同じ列に入らない
        for i in range(3):
            if i == box_row: continue
            candidates[(3*i):(3*i+3),(3*box_col):(3*box_col+3)] &= ~mask[None,:]

        # 同じ箱列に属する自身以外の箱について、各列で候補となっている数字を洗い出す
        mask = numpy.zeros(3, dtype='int32')
        for i in range(3):
            if i == box_row: continue
            box = candidates[(3*i):(3*i+3),(3*box_col):(3*box_col+3)]
            mask |= numpy.array([functools.reduce(lambda x,y: x | y, box[:,i]) for i in range(3)])

        # 自身以外の箱の各列について、その列にのみ含まれない候補を洗い出す
        mask = numpy.array([
            (mask[1] | mask[2]) & ~mask[0],
            (mask[2] | mask[0]) & ~mask[1],
            (mask[0] | mask[1]) & ~mask[2],
        ])

        # 自身以外の箱の各列についてその列にのみ含まれない数字は、自身の箱のその列にしか含まれない
        mask = numpy.array([
            mask[1] | mask[2],
            mask[2] | mask[0],
            mask[0] | mask[1],
        ])
        candidates[(3*box_row):(3*box_row+3),(3*box_col):(3*box_col+3)] &= ~mask[None,:]
    
    # 処理後の候補を返す
    return candidates


# In[ ]:


# ナンプレを解く関数
def solve(table:numpy.ndarray) -> numpy.ndarray:

    # 複数解に対応するため、解の配列を用意する
    answers = []

    # スタックを用意する
    stack = []

    # 候補と各マスで整理が行われたかを格納する配列を用意する
    candidates = table_to_candidates(table)
    arranged = numpy.full(candidates.shape, False)
    stack.append((candidates, arranged))

    # スタックの中身がなくなるまで処理を続ける
    while stack:

        # スタックからテーブルを取得する
        candidates, arranged = stack.pop()

        # まずは理詰めだけで解く
        while True:

            # 処理前の候補を記録
            tmp = candidates.copy()

            # 候補に変化が起きなくなるまで整理を行う
            while True:
                _tmp = candidates.copy()
                candidates, arranged = arrange_candidates(candidates, arranged)
                if numpy.all(candidates==_tmp): break

            # 集合の分割により削除を行う
            candidates = divide_set_row(candidates)
            candidates = divide_set_col(candidates)
            candidates = divide_set_box(candidates)

            # 集合の差分により削除を行う
            candidates = subtract_set_rowbox(candidates)
            candidates = subtract_set_colbox(candidates)

            # それでも候補に変化が起きなくなったらループを止める
            if numpy.all(candidates==tmp): break

        # 矛盾が発生した場合は他に何もしない
        if numpy.any(candidates == 0): continue

        # そうでない場合で候補が残っている場合
        elif numpy.any(count_bits(candidates)!=1):

            # 候補が定まっていないマスを探索し
            row, col = numpy.where(count_bits(candidates)!=1)

            # 各候補を仮置きしてスタックに突っ込む
            candidate = candidates[row[0], col[0]]
            for candidate in [1 << i for i in range(9) if (1 << i & candidate) != 0]:
                tmp_c = candidates.copy()
                tmp_a = arranged.copy()
                tmp_c[row[0], col[0]] = candidate
                stack.append((tmp_c, tmp_a))

        # 候補が残っていない場合は解を変換し、answers に格納する
        else: answers.append(candidates_to_table(candidates))
    
    # 求めた解を返す
    return answers


# In[ ]:


# 解けているかどうかを確認する関数
def check(table:numpy.ndarray) -> numpy.ndarray:

    # 候補が全て埋まっているかをチェック
    if numpy.any(table == 0): return False

    # マスが属する行についてチェック
    for i in range(9):
        numbers = numpy.unique(table[i,:])
        if numbers.shape[0] != 9: return False

    # マスが属する列についてチェック
    for i in range(9):
        numbers = numpy.unique(table[:,i])
        if numbers.shape[0] != 9: return False

    # マスが属する箱についてチェック
    for i, j in itertools.product(range(3), range(3)):
        box_row = 3 * (i // 3)
        box_col = 3 * (j // 3)
        numbers = numpy.unique(table[box_row:(box_row+3),box_col:(box_col+3)])
        if numbers.shape[0] != 9: return False

    # 全て確認が取れたら True を返す
    return True


# In[ ]:


# テーブルを読み込む
tables = [numpy.loadtxt(f'./dataset/problem-{i+1:04}.csv', delimiter=',', dtype='int32') for i in range(5000)]

# アルゴリズムの性能評価
for table in tqdm.tqdm(tables):
    result = solve(table)
    assert(len(result)==1 and check(result[0]))

