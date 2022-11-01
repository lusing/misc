package main

import "runtime"

func test() {
	println("test")
}

func main() {
	go test()
	println(runtime.NumGoroutine())
	var a1 uint8 = 0b000
	a2 := a1 + 1
	println(a2)
}
