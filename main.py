from surprise import Dataset, SVD

data = Dataset.load_builtin("ml-100k", prompt=False)
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

print("Surprise works")