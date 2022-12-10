-module(geometry).
-export([area/1]).

area({circle, R}) -> 3.14159 * R * R;
area({rectangle, W, H}) -> W * H.
