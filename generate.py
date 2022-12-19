import numpy
import sudoku

def generate():

    while True:

        candidates = numpy.full((9,9), 0b111111111, dtype='int32')

        while True:

            if numpy.any(candidates==0):
                candidates = numpy.full((9,9), 0b111111111, dtype='int32')

            row = numpy.random.randint(0, 9)
            col = numpy.random.randint(0, 9)
            candidate = [1 << x for x in range(9) if (1 << x & candidates[row,col]) != 0]
            candidates[row,col] = numpy.random.choice(candidate)

            candidates, finished = sudoku.logic_solve(sudoku.candidates_to_table(candidates))
            if finished: break

        table = candidates
        solvable = True

        while solvable:

            solvable = False

            rows, cols = numpy.where(table!=0)
            index = numpy.arange(rows.shape[0])
            numpy.random.shuffle(index)

            for i in index:

                table_tmp = table.copy()
                table_tmp[rows[i], cols[i]] = 0
                candidates, finished = sudoku.logic_solve(table_tmp)

                if finished:
                    solvable = True
                    table = table_tmp
                    break

        return table

if __name__ == '__main__':
    for i in range(8814, 10000):
        numpy.savetxt(f'./dataset/problem-{i+1:04}.csv', generate(), fmt='%d', delimiter=',')