use std::ops::{AddAssign, Add};
use std::sync::Arc;
use std::{rc::Rc, borrow::Borrow};
use std::fmt::Display;
use std::fmt::Debug;

struct UInt8{
    value: u8
}

impl UInt8{
    fn new(value: u8) -> Self{
        Self{value}
    }
}

impl Display for UInt8{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Debug for UInt8{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl AddAssign for UInt8{
    fn add_assign(&mut self, other: Self){
        self.value += other.value;
    }
}

enum Integers{
    Int8=1,
    Int16,
    Int32,
    Int64,
    Int128,
    Bigint
}

impl Display for Integers{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", (*self as i32))
    }
}

enum Bits8 {
    Uint8(u8),
    Int8(i8)
}

impl Add for Bits8{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Bits8::Uint8(a), Bits8::Uint8(b)) => Bits8::Uint8(a + b),
            (Bits8::Int8(a), Bits8::Int8(b)) => Bits8::Int8(a + b),
            (Bits8::Uint8(a), Bits8::Int8(b)) => Bits8::Int8(a as i8 + b),
            (Bits8::Int8(a), Bits8::Uint8(b)) => Bits8::Int8(a + b as i8),
        }
    }
}

impl Add for &Bits8{
    type Output = Arc<Bits8>;
    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Bits8::Uint8(a), Bits8::Uint8(b)) => Arc::new(Bits8::Uint8(a + b)),
            (Bits8::Int8(a), Bits8::Int8(b)) => Arc::new(Bits8::Int8(a + b)),
            (Bits8::Uint8(a), Bits8::Int8(b)) => Arc::new(Bits8::Int8((*a) as i8 + b)),
            (Bits8::Int8(a), Bits8::Uint8(b)) => Arc::new(Bits8::Int8(a + (*b) as i8)),
        }
    }
}

impl Display for Bits8{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bits8::Uint8(value) => write!(f, "{}", value),
            Bits8::Int8(value) => write!(f, "{}", value)
        }
    }
}

impl Debug for Bits8{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bits8::Uint8(value) => write!(f, "{}", value),
            Bits8::Int8(value) => write!(f, "{}", value)
        }
    }
}

fn test(i:Option<Box<i32>>) -> Result<Box<i32>, &'static str>{
    match i{
        Some(i) => Ok(i),
        None => Err("NAN")
    }
}

fn main() {
    let a1 = Box::new(Some(0u8));
    println!("{:?}",a1);

    let a2 = (*a1).unwrap() + 1u8;
    println!("{}",a2);

    let a3 = (*a1).unwrap() + 2u8;
    println!("{}",a3);

    let b1 = Box::new(4u8);
    println!("{:?}",b1);

    let b2 = (*b1) + 1u8;
    let b3 = (*b1) + 2u8;
    println!("{}",b2);
    println!("{}",b3);

    let c1 = 0u8;
    let c2 = c1 + 1u8;
    let c3 = c1 + 2u8;
    println!("{}",c2);
    println!("{}",c3);

    let i1 = 1i32;
    let i2 = i1 + 1;
    let i3 = i1 + i2;
    println!("{} {} {}",i1,i2,i3);

    let i11 = Rc::new(1i32);
    let i12 = Rc::new(i11.borrow() + 1i32);
    let i13 = i11.borrow() + 2i32;
    println!("{} {} {}",i11,i12,i13);

    let mut u101 = UInt8::new(101u8);
    u101+=UInt8::new(1u8);
    let u103 = UInt8::new(u101.value + 2u8);
    println!("{:?} {}",u101,u103);

    let u201 = Box::new(UInt8::new(11u8));
    let u202 = u201.value + 1u8;
    let u203 = u201.value + 2u8;
    println!("{:?} {} {}",u201,u202,u203);

    let bit1 = Box::new(Bits8::Uint8(31u8));
    let bit2 = Box::new(Bits8::Int8(-32i8));

    match *bit1 {
        Bits8::Uint8(value) => println!("unsinged int:{}",value),
        Bits8::Int8(value) => println!("signed int:{}",value)
    }

    match *bit2 {
        Bits8::Int8(value) => println!("signed int:{}",value),
        _ => {}
    }

    if let Bits8::Uint8(value) = *bit1 {
        println!("unsinged int:{}",value);
    }

    if let Bits8::Uint8(value) = *bit2 {
        println!("unsinged int:{}",value);
    } else {
        println!("not unsinged int");
    }

    let bit3 = *bit1 + *bit2;
    println!("{:}",bit3);

    let int1 = Integers::Int8;
    println!("{:}",int1);

    let num1 = Box::new(Some(1i32));
    let num2 = num1.unwrap() + 1i32;
    if let Some(num3) = *num1 {
        let num4 = num3 + 2i32;
        println!("{} {} {}",num2,num3,num4);
    }
    println!("Num2:{:?}",num2);

    let bit4 = Box::new(Some(Bits8::Uint8(31u8)));
    if let Some(Bits8::Uint8(value)) = *bit4 {
        println!("{:}",value);
    }

    let _ = test(Some(Box::new(1i32)));

    let bit4 = Arc::new(Bits8::Uint8(1));
    let bit5 = Arc::new(Bits8::Int8(-1));
    let b4: &Bits8= bit4.borrow();
    let b5: &Bits8= bit5.borrow();
    let b6 = b4 + b5;
    println!("Arc:{:?}",b6);

    //let bit7 = *Box::new(Bits8::Int8(-2)) + *bit4;
    //println!("bit6:{}",bit6);
    //println!("bit7:{}",bit7);

    // let bit11 = Box::new(Bits8::Uint8(1));
    // let bit12 = Box::new(Bits8::Int8(-1));
    // let bit13 = bit4 + bit5;
    // println!("bit13:{}",bit13);

    // let i001 = Box::new(1i32);
    // let i002 = Box::new(-1i32);
    // let i003 = i001 + 1i32;
    // println!("i003:{}",i003);

}
