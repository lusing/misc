-module(hello).
-export([hello/0]).

hello() ->
    io:format("Hello, world~n").

