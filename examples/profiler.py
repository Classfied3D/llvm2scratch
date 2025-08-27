import cProfile, pstats

cProfile.run("import compile; compile.main()", "profile.out")
p = pstats.Stats("profile.out")
p.strip_dirs().sort_stats("tottime").print_stats(20)
