(module
    (func (param $a i64) (param $b i64) (result i64)
        (i64.rem_u 
            (local.get $a) 
            (local.get $b)
        )
    )
    (export "rem_u_i64" (func 0))
)