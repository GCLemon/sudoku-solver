#!/usr/bin/env python
# coding: utf-8

# # 数独は集合演算だよねってお話

# In[1]:


#必要なライブラリのインポート
import numpy
import numba
import tqdm


# In[2]:


# ビットカウントを行う関数
@numba.vectorize([numba.i4(numba.i4), numba.i8(numba.i8)])
def count_bits(bits:int) -> int:
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555)
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333)
    bits = (bits & 0x0F0F0F0F) + (bits >> 4 & 0x0F0F0F0F)
    bits = (bits & 0x00FF00FF) + (bits >> 8 & 0x00FF00FF)
    bits = (bits & 0x0000FFFF) + (bits >> 16 & 0x0000FFFF)
    return bits


# In[3]:


# 最上位ビットを取得する関数
@numba.vectorize([numba.i4(numba.i4), numba.i8(numba.i8)])
def calc_msb(bits:int) -> int:
    msb = numpy.where(bits & 0xFFFF0000, 16, 0)
    msb += numpy.where(bits & 0xFF00FF00, 8, 0)
    msb += numpy.where(bits & 0xF0F0F0F0, 4, 0)
    msb += numpy.where(bits & 0xCCCCCCCC, 2, 0)
    msb += numpy.where(bits & 0xAAAAAAAA, 1, 0)
    return msb + 1


# In[4]:


# テーブルから候補を生成する関数
@numba.jit(numba.i4[:,:](numba.i4[:,:]))
def table_to_candidates(table:numpy.ndarray) -> numpy.ndarray:
    candidates = numpy.full(table.shape, 0b111111111, dtype='int32')
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if table[i,j] != 0:
                candidates[i,j] = 1 << (table[i,j] - 1)
    return candidates


# In[5]:


# 候補からテーブルを生成する関数
@numba.njit(numba.i4[:,:](numba.i4[:,:]))
def candidates_to_table(candidates:numpy.ndarray) -> numpy.ndarray:
    table = numpy.zeros(candidates.shape, dtype='int32')
    for i in range(candidates.shape[0]):
        for j in range(candidates.shape[1]):
            if count_bits(candidates[i,j]) == 1:
                table[i,j] = calc_msb(candidates[i,j])
    return table


# In[6]:


# 候補を整理する関数
@numba.njit(numba.types.Tuple((numba.i4[:,:],numba.b1[:,:]))(numba.i4[:,:],numba.b1[:,:]))
def arrange_candidates(candidates:numpy.ndarray, arranged:numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:

    # 整理を行っていない各マスに対して処理
    for row, col in zip(*numpy.where(~arranged)):

        # 候補が一つでないときの処理
        if count_bits(candidates[row,col]) != 1:

            # 自身が属している行・列・箱について候補となっている数字を計算
            set_row = 0b000000000
            for i in range(9):
                if i != col:
                    set_row |= candidates[row,i]
                    
            set_col = 0b000000000
            for i in range(9):
                if i != row:
                    set_col |= candidates[i,col]
                    
            set_box = 0b000000000
            for i in range(3 * (row // 3), 3 * (row // 3 + 1)):
                for j in range(3 * (col // 3), 3 * (col // 3 + 1)):
                    if i != row or j != col:
                        set_box |= candidates[i,j]

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
            for i in range(3 * (row // 3), 3 * (row // 3 + 1)):
                for j in range(3 * (col // 3), 3 * (col // 3 + 1)):
                    if i != row or j != col:
                        candidates[i,j] &= ~candidates[row,col]
    
    # 処理後の候補と各マスで整理済みかを返す
    return candidates, arranged


# In[7]:


# 配列の分割を行うジェネレータ
@numba.njit
def divided_array(array:numpy.ndarray):

    # 集合を分割する場合の数を計算
    condnum = 1 << array.shape[0]

    # そのインデックスで計算を飛ばすかを記録
    skip = numpy.full(condnum, False)

    # 各インデックスについて、ビットが0か1かで集合を分割
    for i in range(condnum):
        if skip[i]: continue
        skip[i] = skip[i^(condnum-1)] = True
        mask = i & (1 << numpy.arange(array.shape[0]))
        yield array[mask!=0], array[mask==0]


# In[8]:


# テーブルの各行に対して集合分割を行う関数
@numba.njit(numba.i4[:,:](numba.i4[:,:]))
def divide_set_row(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各行に対して処理
    for row in range(9):

        # 行において、候補が一つに定まっていないマスの集合を取得
        index = numpy.array([x for x in range(9) if count_bits(candidates[row,x]) != 1])
        index = [x for x in divided_array(index)]

        # マスの集合を分割し、それぞれの場合について処理
        for i in range(len(index)):
            index1, index2 = index[i]

            # 分割した集合の要素数が両方2以上の場合に処理
            if index1.shape[0] <= 1: continue
            if index2.shape[0] <= 1: continue

            # マスと数字の集合計算
            set1 = 0b000000000
            for i in range(index1.shape[0]):
                set1 |= candidates[row,index1[i]]
            if count_bits(set1) == index1.shape[0]:
                for i in range(index2.shape[0]):
                    candidates[row,index2[i]] &= ~set1
            
            # マスと数字の集合計算
            set2 = 0b000000000
            for i in range(index2.shape[0]):
                set2 |= candidates[row,index2[i]]
            if count_bits(set2) == index2.shape[0]:
                for i in range(index1.shape[0]):
                    candidates[row,index1[i]] &= ~set2

    # 処理後の候補を返す
    return candidates


# In[9]:


# テーブルの各列に対して集合分割を行う関数
@numba.njit(numba.i4[:,:](numba.i4[:,:]))
def divide_set_col(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各列に対して処理
    for col in range(9):

        # 列において、候補が一つに定まっていないマスの集合を取得
        index = numpy.array([x for x in range(9) if count_bits(candidates[x,col]) != 1])
        index = [x for x in divided_array(index)]

        # マスの集合を分割し、それぞれの場合について処理
        for i in range(len(index)):
            index1, index2 = index[i]

            # 分割した集合の要素数が両方2以上の場合に処理
            if index1.shape[0] <= 1: continue
            if index2.shape[0] <= 1: continue

            # マスと数字の集合計算
            set1 = 0b000000000
            for i in range(index1.shape[0]):
                set1 |= candidates[index1[i],col]
            if count_bits(set1) == index1.shape[0]:
                for i in range(index2.shape[0]):
                    candidates[index2[i],col] &= ~set1
            
            # マスと数字の集合計算
            set2 = 0b000000000
            for i in range(index2.shape[0]):
                set2 |= candidates[index2[i],col]
            if count_bits(set2) == index2.shape[0]:
                for i in range(index1.shape[0]):
                    candidates[index1[i],col] &= ~set2

    # 処理後の候補を返す
    return candidates


# In[10]:


# テーブルの各箱に対して集合分割を行う関数
@numba.njit(numba.i4[:,:](numba.i4[:,:]))
def divide_set_box(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各箱に対して処理
    for box_row in range(3):
        for box_col in range(3):

            # 列において、候補が一つに定まっていないマスの集合を取得
            index = numpy.array([
                y * 3 + x
                for y in range(3)
                for x in range(3)
                if count_bits(candidates[3*box_row+y,3*box_col+x]) != 1
            ])
            index = [x for x in divided_array(index)]

            # マスの集合を分割し、それぞれの場合について処理
            for i in range(len(index)):
                index1, index2 = index[i]

                # 分割した集合の要素数が両方2以上の場合に処理
                if index1.shape[0] <= 1: continue
                if index2.shape[0] <= 1: continue

                # インデックスを変換
                box_row1 = index1 // 3 + 3 * box_row
                box_col1 = index1 % 3 + 3 * box_col
                box_row2 = index2 // 3 + 3 * box_row
                box_col2 = index2 % 3 + 3 * box_col

                # マスと数字の集合計算
                set1 = 0b000000000
                for i in range(box_row1.shape[0]):
                    set1 |= candidates[box_row1[i],box_col1[i]]
                if count_bits(set1) == index1.shape[0]:
                    for i in range(box_row2.shape[0]):
                        candidates[box_row2[i],box_col2[i]] &= ~set1
                
                # マスと数字の集合計算
                set2 = 0b000000000
                for i in range(box_row2.shape[0]):
                    set2 |= candidates[box_row2[i],box_col2[i]]
                if count_bits(set2) == index2.shape[0]:
                    for i in range(box_row1.shape[0]):
                        candidates[box_row1[i],box_col1[i]] &= ~set2

    # 処理後の候補を返す
    return candidates


# In[11]:


# テーブルの各箱行に対して集合差分を行う関数
@numba.njit(numba.i4[:,:](numba.i4[:,:]))
def subtract_set_rowbox(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各箱に対して処理
    for box_row in range(3):
        for box_col in range(3):

            # 箱の内容を取得
            box = candidates[(3*box_row):(3*box_row+3),(3*box_col):(3*box_col+3)]

            # 箱の中の各行について、その行にしか含まれない候補を洗い出す
            mask = box[:,0] | box[:,1] | box[:,2]
            mask = numpy.array([
                mask[0] & ~(mask[1] | mask[2]),
                mask[1] & ~(mask[2] | mask[0]),
                mask[2] & ~(mask[0] | mask[1]),
            ])

            # 先ほどの候補は、同じ箱行に属する箱において同じ行に入らない
            for i in range(3):
                if i == box_col: continue
                for y in range(3):
                    for x in range(3):
                        candidates[3*box_row+y,3*i+x] &= ~mask[y]

            # 同じ箱行に属する自身以外の箱について、各行で候補となっている数字を洗い出す
            mask = numpy.zeros(3, dtype='int32')
            for i in range(3):
                if i == box_col: continue
                box = candidates[(3*box_row):(3*box_row+3),(3*i):(3*i+3)]
                mask |= box[:,0] | box[:,1] | box[:,2]

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
            for y in range(3):
                for x in range(3):
                    candidates[3*box_row+y,3*box_col+x] &= ~mask[y]
    
    # 処理後の候補を返す
    return candidates


# In[12]:


# テーブルの各箱列に対して集合差分を行う関数
@numba.njit(numba.i4[:,:](numba.i4[:,:]))
def subtract_set_colbox(candidates:numpy.ndarray) -> numpy.ndarray:

    # 各箱に対して処理
    for box_row in range(3):
        for box_col in range(3):

            # 箱の内容を取得
            box = candidates[(3*box_row):(3*box_row+3),(3*box_col):(3*box_col+3)]

            # 箱の中の各列について、その列にしか含まれない候補を洗い出す
            mask = box[0,:] | box[1,:] | box[2,:]
            mask = numpy.array([
                mask[0] & ~(mask[1] | mask[2]),
                mask[1] & ~(mask[2] | mask[0]),
                mask[2] & ~(mask[0] | mask[1]),
            ])

            # 先ほどの候補は、同じ箱列に属する箱において同じ列に入らない
            for i in range(3):
                if i == box_row: continue
                for y in range(3):
                    for x in range(3):
                        candidates[3*i+y,3*box_col+x] &= ~mask[x]

            # 同じ箱列に属する自身以外の箱について、各列で候補となっている数字を洗い出す
            mask = numpy.zeros(3, dtype='int32')
            for i in range(3):
                if i == box_row: continue
                box = candidates[(3*i):(3*i+3),(3*box_col):(3*box_col+3)]
                mask |= box[0,:] | box[1,:] | box[2,:]

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
            for y in range(3):
                for x in range(3):
                    candidates[3*box_row+y,3*box_col+x] &= ~mask[x]
    
    # 処理後の候補を返す
    return candidates


# In[13]:


# ナンプレを解く関数
@numba.njit(numba.types.List(numba.i4[:,:])(numba.i4[:,:]))
def solve(table:numpy.ndarray) -> list[numpy.ndarray]:

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


# In[14]:


# 理詰めだけでナンプレを解く関数
@numba.njit(numba.types.Tuple((numba.i4[:,:],numba.b1))(numba.i4[:,:]))
def logic_solve(table:numpy.ndarray) -> tuple[numpy.ndarray,bool]:

    # 候補と各マスで整理が行われたかを格納する配列を用意する
    candidates = table_to_candidates(table)
    arranged = numpy.full(candidates.shape, False)

    # 理詰めだけで解く
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

    # 候補が定まりきらなかった場合は候補を返す
    if numpy.any(count_bits(candidates)!=1): return candidates, False
    
    # そうでない場合は求めた解を返す
    else: return candidates_to_table(candidates), True


# In[15]:


# 解けているかどうかを確認する関数
import itertools

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


# In[16]:


if __name__ == '__main__':
    
    # テーブルを読み込む
    tables = [numpy.loadtxt(f'./dataset/problem-{i+1:04}.csv', delimiter=',', dtype='int32') for i in range(5000)]

    # アルゴリズムの性能評価
    for table in tqdm.tqdm(tables):
        result = solve(table)
        assert(len(result)==1 and check(result[0]))


# In[17]:


if __name__ == '__main__':

    table = numpy.array([
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ], dtype='int32')

    print(solve(table))

