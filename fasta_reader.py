import pandas as pd
import csv
from Bio import Seq, SeqIO

def format(gbff, fna, output_file):
    fasta_index = SeqIO.index(fna, "fasta")

    with open(output_file, 'w') as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(["plasmid_id", "contig"])

        for record in SeqIO.parse(gbff, "genbank"):
            plasmid_id = record.id  # Get the first word as ID
            if plasmid_id not in fasta_index:
                print(f"Warning: {plasmid_id} not found in FASTA file, skipping.")
                continue
            record.seq = fasta_index[plasmid_id].seq  # Update the record's sequence with the FASTA sequence
            contig_parts = []
            last_end = 0
            cds_features = [f for f in record.features if f.type == "CDS"]
            cds_features.sort(key=lambda f: int(f.location.start))
            for feature in cds_features:
                start = int(feature.location.start)
                end = int(feature.location.end)
                strand = feature.location.strand
                protein = feature.qualifiers.get("translation", [""])[0]  # Get the protein sequence if available
                if not protein:
                    cds_seq = feature.location.extract(record.seq)
                    remainder = len(cds_seq) % 3
                    if remainder != 0:
                        cds_seq += "N" * (3 - remainder) # Pad with 'N' to make length a multiple of 3
                    protein = str(Seq.Seq(cds_seq).translate(to_stop=True))
                protein = protein.replace("*", "")  # Remove stop codons from the protein sequence
                if start > last_end: #is there an intergenic region before this CDS?
                    intergenic = record.seq[last_end:start].lower()
                    if intergenic:
                        contig_parts.append("<+>" + str(intergenic))
                strand_marker = "<+>" if strand == 1 else "<->"
                contig_parts.append(strand_marker + protein.upper())
                last_end = end

            if last_end < len(record.seq): #is there an intergenic region after the last CDS?
                intergenic = record.seq[last_end:].lower()
                if intergenic:
                    contig_parts.append("<+>" + str(intergenic))

            writer.writerow([plasmid_id, "".join(contig_parts)])

                
if __name__ == "__main__":
    format("datasets/plasmid.genomic.gbff/plasmid.genomic.gbff", "datasets/plasmid.genomic.fna/plasmid.genomic.fna", "ref_seq_plasmids.csv")
    print(pd.read_csv("ref_seq_plasmids.csv").head(5))