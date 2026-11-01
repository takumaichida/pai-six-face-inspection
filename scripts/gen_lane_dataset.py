import sys, time
from multiprocessing import Pool
sys.path.insert(0, '.')

def job(args):
    from sortbot.lane_dataset import generate
    n, seed, path = args
    return generate(n, seed, path)

if __name__ == '__main__':
    n_per, workers = int(sys.argv[1]), int(sys.argv[2])
    t0 = time.time()
    jobs = [(n_per, 2000 + w, f'data/lane_train_{w}.npz') for w in range(workers)]
    jobs += [(n_per // 3, 8000 + w, f'data/lane_test_{w}.npz') for w in range(2)]
    with Pool(len(jobs)) as p:
        p.map(job, jobs)
    print('done', round(time.time() - t0, 1), 's')
