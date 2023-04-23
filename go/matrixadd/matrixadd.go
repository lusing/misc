package main

import (
	"fmt"
	"sync"
)

func matrixAddWorker(A, B, C [][]int, startRow, endRow int, wg *sync.WaitGroup) {
	for i := startRow; i < endRow; i++ {
		for j := 0; j < len(A[0]); j++ {
			C[i][j] = A[i][j] + B[i][j]
		}
	}
	wg.Done()
}

func matrixAdd(A, B [][]int) [][]int {
	rows := len(A)
	cols := len(A[0])

	C := make([][]int, rows)
	for i := range C {
		C[i] = make([]int, cols)
	}

	var wg sync.WaitGroup

	workerCount := 4
	rowsPerWorker := rows / workerCount

	for i := 0; i < workerCount; i++ {
		startRow := i * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if i == workerCount-1 {
			endRow = rows
		}
		wg.Add(1)
		go matrixAddWorker(A, B, C, startRow, endRow, &wg)
	}

	wg.Wait()
	return C
}

func main() {
	A := [][]int{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9}}

	B := [][]int{
		{9, 8, 7},
		{6, 5, 4},
		{3, 2, 1},
	}
	C := matrixAdd(A, B)
	for _, row := range C {
		fmt.Println(row)
	}
}
