(module
    (func (param $a i32) (result i32)
        (i32.add 
            (local.get $a) 
            (i32.const 1)
        )
    )
    (export "inc_i32" (func 0))
)