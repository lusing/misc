(module
    (func (param $a f32) (result f32)
        (f32.mul 
            (local.get $a) 
            (f32.const 1024)
        )
    )
    (export "mul_1k_f32" (func 0))
)