# Read data from file
inputfile = "dna.txt"
f = open(inputfile, 'r')
seq = f.read()

# read file using with statement
inputfile = "dna.tx"
with open("dna.txt", 'r') as f:
    seq = f.read()
seq = seq.replace("\n", "")
seq = seq.replace("\r", "")

# print(seq).txt
# make function to read data


def read_seq(inputfile):
    """Function read and save sequence with special character removed"""

    with open(inputfile, 'r') as f:
        seq = f.read()
        # to remove extra line beaks given by \n
        seq = seq.replace("\n", "")
        # sometime we have unother character hiding in the string and than can be replced by "\r "
# This is extra step just to be on safer side
        seq = seq.replace("\r", "")
    return seq


# print(seq)


def translate(seq):
    # Write Doc string so user understande te class or function
    """Translate a string containg nucleotide sequnce into a string
    containg the corresponding sequence of amino acids. Nucleotides are 
    translated in to triplets and the table dictionary; each amino acid
    is encoded with a string of length 1."""
    table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }

    # check that sequence length is divisible by 3
    n = len(seq) % 3  # '%' is called as modulur
    print(n)

    protine = ""
    # loop over the sequence
    if n == 0:
        for i in range(0, len(seq), 3):
            # extract the single codon
            codon = seq[i:i+3]
            # look up the codon and store the results
            protine += table[codon]  # concatinate the string
        return(protine)


# to check the function to extraxt protine
print(translate("CTA"))
help(translate)
dna = read_seq("dna.txt")
prt_f = translate(dna[20:938])[:-1]
prt = read_seq("protine.txt")
print(prt_f)
print(prt)
print(prt_f == prt)
# to avoid stop codon for comparision
#print((prt_f[:-1]) == prt)
# or we can d the same by using[20:935] for print_f
