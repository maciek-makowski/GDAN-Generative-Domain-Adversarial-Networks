from do_mpc.data import save_results, load_results

results = load_results('./results/results.pkl')

print("results", results)
keys = ['_x', '_u', '_p', '_tvp', '_z', '_aux']


print("Results MPC", results['mpc'])
print("Results simulation", results['simulator'])
for key in keys:
    #print(key, results['mpc'][key])
    #if key == '_tvp':
    print(key)
    for i in results['mpc'][key]:
            print(i)


print(" NOW          SIMULATOR       ")

for key in keys:
    #print(key, results['simulator'][key])
    #if key == '_p':
    print(key)
    for i in results['simulator'][key]:
        print(i)
