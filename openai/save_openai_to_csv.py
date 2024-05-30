import csv

def save_data_to_csv(synthetic_results, filename="synthetic_data.csv"):
    """
    Saves synthetic data to a CSV file.
    The field names are based on the pydantic object. You can customize it based on the data you want to generate
    """

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['patient_id', 'patient_name', 'diagnosis_code', 'procedure_code', 'total_charge', 'insurance_claim_amount']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in synthetic_results:
            writer.writerow(result.dict())

    print(f"Data saved to {filename}")