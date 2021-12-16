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

v8::MaybeLocal<v8::String> ReadFile(v8::Isolate *isolate, const char *name) {
  FILE *file = fopen(name, "rb");
  if (file == NULL)
    return v8::MaybeLocal<v8::String>();

  fseek(file, 0, SEEK_END);
  size_t size = ftell(file);
  rewind(file);

  char *chars = new char[size + 1];
  chars[size] = '\0';
  for (size_t i = 0; i < size;) {
    i += fread(&chars[i], 1, size - i, file);
    if (ferror(file)) {
      fclose(file);
      return v8::MaybeLocal<v8::String>();
    }
  }
  fclose(file);
  v8::MaybeLocal<v8::String> result = v8::String::NewFromUtf8(
      isolate, chars, v8::NewStringType::kNormal, static_cast<int>(size));
  delete[] chars;
  return result;
}


int main(int argc, char *argv[]) {
  char *filename = "./script.js";
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
      v8::Local<v8::String> source;
      if (!ReadFile(isolate, filename).ToLocal(&source)) {
        printf("Cannot read from %s\n", filename);
      }
      v8::Local<v8::Script> script =
          v8::Script::Compile(context, source).ToLocalChecked();
      v8::Local<v8::Value> result = script->Run(context).ToLocalChecked();
      v8::String::Utf8Value utf8(isolate, result);
      printf("%s\n", *utf8);
      printf("%s\n", v8::V8::GetVersion());
    }
  }
  isolate->Dispose();
  v8::V8::Dispose();
  v8::V8::ShutdownPlatform();
  delete create_params.array_buffer_allocator;
  return 0;
}
