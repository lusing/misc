-module(geometry).
-export([area/1]).

area({circle,R}) -> pi*R*R;
area({rectangle,W,H}) -> W*H.
