#include "src/pch.h"

std::filesystem::path get_out_dir(std::string_view file)
{
  std::filesystem::path result(file);
  result.replace_extension("");

  result = std::format("{}/output/{}/", //
    result.parent_path().c_str(), result.filename().c_str());
  return result;
}
