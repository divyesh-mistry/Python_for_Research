# Read data from file
inputfile = "dna.txt"
f = open(inputfile, 'r')
seq = f.read()
print(seq)

# to remove extra spliting of lines given by \n
seq = seq.replace("\n", "")
print(seq)
