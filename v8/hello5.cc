#include "include/libplatform/libplatform.h"
#include "include/v8-context.h"
#include "include/v8-initialization.h"
#include "include/v8-isolate.h"
#include "include/v8-local-handle.h"
#include "include/v8-primitive.h"
#include "include/v8-script.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  v8::V8::InitializeICUDefaultLocation(argv[0]);
  v8::V8::InitializeExternalStartupData(argv[0]);
  std::unique_ptr<v8::Platform> platform = v8::platform::NewDefaultPlatform();
  v8::V8::InitializePlatform(platform.get());
  v8::V8::Initialize();
  v8::Isolate::CreateParams create_params;
  create_params.array_buffer_allocator =
      v8::ArrayBuffer::Allocator::NewDefaultAllocator();
  v8::Isolate *isolate = v8::Isolate::New(create_params);
  {
    v8::Isolate::Scope isolate_scope(isolate);
    v8::HandleScope handle_scope(isolate);
    v8::Local<v8::Context> context = v8::Context::New(isolate);
    v8::Context::Scope context_scope(context);
    {
      v8::Local<v8::String> source = v8::String::NewFromUtf8Literal(
          isolate, "let f1 = (x) => x*x; f1(1.2);");
      v8::Local<v8::Script> script =
          v8::Script::Compile(context, source).ToLocalChecked();
      v8::Local<v8::Value> result = script->Run(context).ToLocalChecked();
      v8::String::Utf8Value utf8(isolate, result);
      printf("%s\n", *utf8);
      char *str1 = "hello,v8";
      v8::MaybeLocal<v8::String> v8str1 =
          v8::String::NewFromUtf8(isolate, str1, v8::NewStringType::kNormal, strlen(str1));
      v8::String::Utf8Value v8str2(isolate, v8str1.ToLocalChecked());
      printf("%s\n", *v8str2);
      printf("%s\n", v8::V8::GetVersion());
    }
  }
  isolate->Dispose();
  v8::V8::Dispose();
  v8::V8::ShutdownPlatform();
  delete create_params.array_buffer_allocator;
  return 0;
}
