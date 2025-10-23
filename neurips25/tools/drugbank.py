import os
import json
from pathlib import Path
import requests
import pandas as pd
import collections
from requests.auth import HTTPBasicAuth
import zipfile
import xml.etree.ElementTree as ET

URL = "https://go.drugbank.com/releases/5-1-13/downloads/all-full-database"
USERNAME = os.getenv("DRUGBANK_USERNAME", None)
PASSWORD = os.getenv("DRUGBANK_PASSWORD", None)

class DrugBank:
    def __init__(self, local: bool = True, out_path: str = "data/drugbank") -> None:
        if local:
            self.out_path = out_path
            if os.path.exists(os.path.join(out_path, "drugbank.csv")):
                print("Loading data, already preprocessed.")
                self.data = pd.read_csv(os.path.join(out_path, "drugbank.csv"))
            else:
                self.data = None
                self.download()
                self.preprocess() # loads data into self.data
        elif not local:
            raise NotImplementedError("API is not implemented.")
            if USERNAME is None or PASSWORD is None:
                raise ValueError("Please set the DRUGBANK_USERNAME and DRUGBANK_PASSWORD environment variables")
        self.local = local

    def download(self) -> None:
        """
        Automated download.
        Requires setting env variables DRUGBANK_USERNAME and DRUGBANK_PASSWORD
        """
        os.makedirs(self.out_path, exist_ok=True)
        write_path = Path(os.path.join(self.out_path, "all-full-database.zip"))

        if os.path.exists(write_path):
            print("File already downloaded.")
            self.load(write_path)
            return
        
        if USERNAME is None or PASSWORD is None:
            raise ValueError("Please set the DRUGBANK_USERNAME and DRUGBANK_PASSWORD environment variables")
        
        # else download
        print("Downloading...")
        response = requests.get(URL, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        response.raise_for_status()

        if response.status_code == 200:
            with open(write_path, 'wb') as file:
                file.write(response.content)
            print("Download completed successfully.")
            self.load(write_path)
        else:
            print(f"Failed to download the file: \n{response}")
        return
    
    def load(self, path: Path) -> None:
        if self.data is not None:
            print("Data already loaded.")
            return
        if not os.path.exists(os.path.join(path.parent, "full database.xml")):
            print("Extracting the zip file...")
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall("data/drugbank")
        print("Building the tree...")
        self.data = ET.parse('data/drugbank/full database.xml').getroot()
        print("Tree built.")

    def preprocess(self) -> None:
        """
        Preprocesses the DrugBank data.
        Part of this code is inspired from https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb
        """
        print("Pre-processing...")
        ns = '{http://www.drugbank.ca}'
        rows = []
        for i, drug in enumerate(self.data):
            row = collections.OrderedDict()
            assert drug.tag == ns + 'drug'
            row['type'] = drug.get('type')
            row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
            row['name'] = drug.findtext(ns + "name")
            row['description'] = drug.findtext(ns + "description")
            row['groups'] = [group.text for group in
                drug.findall("{ns}groups/{ns}group".format(ns = ns))]
            row['atc_codes'] = [code.get('code') for code in
                drug.findall("{ns}atc-codes/{ns}atc-code".format(ns = ns))]
            row['categories'] = [x.findtext(ns + 'category') for x in
                drug.findall("{ns}categories/{ns}category".format(ns = ns))]
            
            aliases = {
                elem.text for elem in 
                drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
                drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns = ns)) +
                drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
                drug.findall("{ns}products/{ns}product/{ns}name".format(ns = ns))
            }
            aliases.add(row['name'])
            row['aliases'] = sorted(aliases)

            rows.append(row)

        alias_dict = {row['drugbank_id']: row['aliases'] for row in rows}
        with open(os.path.join(self.out_path, 'aliases.json'), 'w') as fp:
            json.dump(alias_dict, fp, indent=2, sort_keys=True)

        def collapse_list_values(row):
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = '|'.join(value)
            return row

        rows = list(map(collapse_list_values, rows))
        columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'description']
        drugbank_df = pd.DataFrame.from_dict(rows)[columns]
        self.data = drugbank_df
        drugbank_df.to_csv(os.path.join(self.out_path, 'drugbank.csv'), index=False)
        print(f"Data saved to {os.path.join(self.out_path, 'drugbank.csv')}. \nDone.")

    def local_query(self, query: str) -> pd.DataFrame:
        """
        Searches the local DrugBank dataset for a drug by its name.

        Args:
            query (str): The name of the drug to search for.

        Returns:
            pd.DataFrame: A Dataframe containing name, type, groups, atc codes, categories, and description
            of the drug.
        """
        return self.data[self.data["name"].str.lower() == query.lower()]
    
    
    def distant_query(self, method: str, query: str) -> str:
        """
        Doc: https://dev.drugbank.com/guides/implementation/medication_search
        """
        raise NotImplementedError("Use local query")
    
    def query(self, **kwargs) -> None:
        if self.local:
            return self.local_query(kwargs.get("query"))
        else:
            return self.distant_query(**kwargs)
        
    def get_all_drugnames(self) -> list:
        """
        Returns a list of all drug names in the DrugBank dataset.
        """
        return self.data["name"].tolist()


if __name__ == "__main__":
    import yaml
    with open("neurips25/configs/base.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    out_path = config["tools"]["drugbank"]
    drugbank = DrugBank(local=True, out_path=out_path)
    print(drugbank.query(method="drug-search", query="capecitabine").name.values[0])
    print(drugbank.query(method="drug-search", query="capecitabine").description.values[0])
    print(drugbank.get_all_drugnames())
