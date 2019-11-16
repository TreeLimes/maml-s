import os



l = [f for f in os.listdir("./") if not f.startswith('.')]



for j, subdir in enumerate(l):

    files = [f for f in os.listdir("./"+subdir) if not f.startswith('.')]

    s = subdir + "/"
    for i, f in enumerate(files):
        print(s+f)

        os.rename(s+f, s + str(i) + ".jpg")
    os.rename("./" + subdir, str(j))
