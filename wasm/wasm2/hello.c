#include <stdio.h>

int fn(int a){
    return 0;
}

int fn2(int a,int b){
    return a+b;
}

int fn3(int a,int b){
    return fn2(a,b)+fn(a);
}

int main(){
    int result = fn3(1,2);
    printf("WebAssembly %d\n",result);
    return result;
}