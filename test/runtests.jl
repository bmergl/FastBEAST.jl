using TestItemRunner

@run_package_tests filter=ti->(:minimal in ti.tags) verbose=true