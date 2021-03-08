#define BOOST_TEST_MODULE onnx_model_test
#include <boost/test/unit_test.hpp>

#include <Acts/Plugins/Onnx/OnnxModel.hpp>

#include <boost/filesystem.hpp>

using namespace boost::unit_test;

class CmdArgs {
  static std::string s_model_path;

 public:
  CmdArgs() {
    auto argc = framework::master_test_suite().argc;
    auto argv = framework::master_test_suite().argv;
    std::vector<std::string> args(argv, argv + argc);

    auto found = std::find(args.begin(), args.end(), "--model_path");

    if (found != args.end() && std::next(found) != args.end())
      s_model_path = *std::next(found);

    std::stringstream err_msg;
    err_msg << "The path '" << s_model_path
            << "' does not exist. Pass the correct path as '--model_path "
               "<path>' to the test.";

    BOOST_TEST_REQUIRE(boost::filesystem::exists(s_model_path), err_msg.str());
  }

  static const auto &model_path() { return s_model_path; }
};

std::string CmdArgs::s_model_path = "test_model.onnx";

BOOST_TEST_GLOBAL_FIXTURE(CmdArgs);

BOOST_AUTO_TEST_CASE(test_and_benchmark_onnx_model) {
  // Environment and options
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark_a");
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  Ort::AllocatorWithDefaultOptions allocator;

  // Segmentation fault for N >= 512
  //     const std::size_t N = 256;
  for (auto N = 1ul; N < 500; N *= 2) {
    using MyModel = Acts::OnnxModel<Acts::OnnxInputs<5>, Acts::OnnxOutputs<5>>;
    MyModel model(env, opts, CmdArgs::model_path());

    std::cout << "N = " << N << std::endl;

    // Method 1:
    auto method_1 = [&]() {
      std::vector<MyModel::InTuple> input_data(N);
      std::vector<MyModel::OutTuple> output_data(N);

      for (auto i = 0ul; i < N; ++i)
        input_data[i] =
            std::tuple_element_t<0, MyModel::InTuple>::Constant(1.f);

      auto t0 = std::chrono::high_resolution_clock::now();

      for (auto i = 0ul; i < N; ++i)
        output_data[i] = model.predict(input_data[i]);

      auto t1 = std::chrono::high_resolution_clock::now();

      const auto ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();

      return std::make_tuple(output_data, ms);
    };

    // Method 2:
    auto method_2 = [&]() {
      MyModel::InVectorTuple input_batch;
      std::get<0>(input_batch).resize(N);

      for (auto i = 0ul; i < N; ++i)
        std::get<0>(input_batch)[i] =
            std::tuple_element_t<0, MyModel::InTuple>::Constant(1.f);

      auto t0 = std::chrono::high_resolution_clock::now();

      const auto output_data_2 = model.predict(input_batch);

      auto t1 = std::chrono::high_resolution_clock::now();

      const auto ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();

      return std::make_tuple(output_data_2, ms);
    };

    const auto [output_1, ms1] = method_1();
    const auto [output_2, ms2] = method_2();

    std::cout << "- Method #1 (individual samples): " << ms1
              << " ms (per input: " << ms1 / static_cast<double>(N) << " ms)\n";
    std::cout << "- Method #2 (batch computation):  " << ms2
              << " ms (per input: " << ms2 / static_cast<double>(N) << " ms)\n";
    std::cout << "- Speedup: " << ms1 / ms2 << "\n";
    std::cout << std::endl;

    for (auto i = 0ul; i < output_1.size(); ++i)
      BOOST_TEST_REQUIRE(std::get<0>(output_1[i]) == std::get<0>(output_2)[i]);
  }
}
