-module(shop1).
-export([total/1]).

total([{What, N} | T]) -> What * N + total(T);
total([]) -> 0.
