import _tagger

a = _tagger.Tagger(90, 2, 32, 0.01, "../train.tsv", "../dev.tsv", "../test.tsv", "../word2vec/GoogleNews-vectors-negative300.txt")
a.train()
b = a.test("CHINA DAILY SAYS VERMIN EAT 7-12 PCT GRAIN STOCKS. A survey of 19 provinces and seven cities showed vermin consume between seven and 12 pct of China's grain stocks, the China Daily said. It also said that each year 1.575 mln tonnes, or 25 pct, of China's fruit output are left to rot, and 2.1 mln tonnes, or up to 30 pct, of its vegetables. The paper blamed the waste on inadequate storage and bad preservation methods. It said the government had launched a national programme to reduce waste, calling for improved technology in storage and preservation, and greater production of additives. The paper gave no further details. REUTER")
print(b)