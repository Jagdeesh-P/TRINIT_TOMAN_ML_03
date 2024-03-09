import pandas as pd

# Load the dataset
with open("../../train.csv", "r") as file:
    lines = file.readlines()

# Remove empty rows
lines = [line.strip() for line in lines if line.strip()]

# Split lines into columns
data = [line.split(",", maxsplit=2) for line in lines]

# Convert to DataFrame
df = pd.DataFrame(data[1:], columns=["filename", "caption", "image"])

# Handle missing values
df.dropna(subset=["filename", "caption", "image"], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove extra spaces
df["filename"] = df["filename"].str.strip()
df["image"] = df["image"].str.strip()
df["caption"] = df["caption"].str.strip()

# Save the cleaned dataset
df.to_csv("cleaned_train.csv", index=False)
