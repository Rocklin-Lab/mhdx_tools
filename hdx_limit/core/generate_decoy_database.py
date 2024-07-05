from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
import numpy as np
import pandas as pd
import argparse

from Bio.SeqUtils import molecular_weight

def calculate_monoisotopic_mass(sequence):
    return molecular_weight(sequence, seq_type='protein', monoisotopic=True)

def generate_random_seq(length):
    """
    length : length of desired sequence

    return a random sequence with the specified length
    """
    aa = ['G', 'A', 'V', 'I', 'L', 'W', 'F', 'Y', 'P', 'S', 'T', 'D', 'E', 'K', 'R', 'H', 'C', 'M', 'Q', 'N']

    return ''.join(random.choice(aa) for _ in range(length))

def generate_unique_seq_df(N,
                           length_min,
                           length_max,
                           mass_min,
                           mass_max,
                           exclusion_mass_list=[0],
                           ppm_tol=50):

    """
    N: number of sequences to be generated
    length_min : minimum sequence length
    length_max : maximum sequence length
    mass_min : minimum mass
    mass_max : maximum mass
    exclusion_mass_list : list of masses to exclude when generating new sequences
    ppm_tol : how difference masses need to be from everything else

    return a dataframe with sequences and MW
    """

    seq_l = []

    i = 0

    while len(seq_l) < N:

        length = random.randint(length_min, length_max)
        seq_tmp = generate_random_seq(length=length)
        mono_mass_tmp = ProteinAnalysis(seq_tmp, monoisotopic=True).molecular_weight()
        if mono_mass_tmp > mass_min and mono_mass_tmp < mass_max:
            differences = np.array([1e6*abs(mono_mass_tmp - i)/mono_mass_tmp for i in exclusion_mass_list])

            if np.min(differences[np.nonzero(differences)]) > ppm_tol:
                print('%i out of %i generated'%(i, N))
                seq_l.append(['decoy_%i'%i, seq_tmp, mono_mass_tmp])
                exclusion_mass_list.append(mono_mass_tmp)
                i += 1

    df = pd.DataFrame(seq_l, columns=['name', 'sequence', 'MW'])

    return df


def main(name_mass_seq,
         decoy_size=1,
         output_path=None,
         mass_min=None,
         mass_max=None,
         length_min=None,
         length_max=None,
         ppm_tol=50): 

    print("HERE")

    df = pd.read_csv(name_mass_seq, names=["name", "sequence", "MW"], skiprows=[0])

    # Check if the "MW" column is defined
    if df["MW"].isnull().all():
        # Compute the monoisotopic mass for each sequence
        df["MW"] = df["sequence"].apply(calculate_monoisotopic_mass)

    if mass_min is None:
        mass_min = min(df['MW'].values) - 50
    if mass_max is None:
        mass_max = max(df['MW'].values) + 50
    if length_min is None:
        length_min = min([len(i) for i in df['sequence'].values]) - 5
    if length_max is None:
        length_max = max([len(i) for i in df['sequence'].values]) + 5

    exclusion_list = df['MW'].values.tolist()

    N = decoy_size * len(df)

    new_df = generate_unique_seq_df(N=N,
                                    mass_min=mass_min,
                                    mass_max=mass_max,
                                    length_min=length_min,
                                    length_max=length_max,
                                    exclusion_mass_list=exclusion_list,
                                    ppm_tol=ppm_tol)

    new_df = pd.concat([df, new_df], ignore_index=True)

    if output_path is None:
        return new_df
    else:
        new_df.to_csv(output_path, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=
        "Generate database with decoys"
    )

    parser.add_argument("-s",
                        "--name_mass_seq",
                        help="path/to/name_mass_seq csv file")
    parser.add_argument("-n",
                        "--decoy_size",
                        default=2,
                        type=int,
                        help="size of decoys as a function of initial protein set")
    parser.add_argument("-o",
                        "--output_path",
                        default=None,
                        help="Output csv file name")
    parser.add_argument("-i",
                        "--mass_min",
                        type=int,
                        default=None,
                        help="Min protein mass")
    parser.add_argument("-f",
                        "--mass_max",
                        default=None,
                        type=int,
                        help="Max protein mass")
    parser.add_argument("-e",
                        "--ppm_tol",
                        default=50,
                        type=float,
                        help="Min ppm distance")

    args = parser.parse_args()

    main(name_mass_seq=args.name_mass_seq,
         decoy_size=args.decoy_size,
         output_path=args.output_path,
         mass_min=args.mass_min,
         mass_max=args.mass_max,
         ppm_tol=args.ppm_tol)
