package main

import "runtime"

func test() {
	println("test")
}

func main() {
	go test()
	println(runtime.NumGoroutine())
}
