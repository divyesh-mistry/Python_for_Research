# Read data from file
inputfile = "dna.txt"
f = open(inputfile, 'r')
seq = f.read()
print(seq)

# to remove extra line beaks given by \n
seq = seq.replace("\n", "")
print(seq)

# sometime we have unother character hiding in the string and than can be replced by "\r "
#This is extra step just to be on safer side
seq = seq.replace("\r", "")
print(seq)
