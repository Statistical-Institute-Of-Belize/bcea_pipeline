# BCEA Classification API

This API provides endpoints for predicting BCEA (Business Classification with Economic Activity) codes from business names and descriptions.

## Starting the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python api_server.py
```

The API server will run at `http://localhost:8000`. Documentation is available at `http://localhost:8000/docs`.

## API Endpoints

### Health Check

```
GET /health
```

Returns the status of the API and model.

### Predict Single Business

```
POST /predict/business
```

Request body:
```json
{
  "bus_name": "Acme Widget Manufacturing",
  "description": "Manufacturing of metal widgets and gadgets for industrial use"
}
```

Response:
```json
{
  "bus_name": "Acme Widget Manufacturing",
  "description": "Manufacturing of metal widgets and gadgets for industrial use",
  "predicted_code": "2599.0",
  "industry_description": "Manufacture of other fabricated metal products n.e.c.",
  "confidence": 0.87,
  "confidence_grade": "high",
  "alternatives": [
    {
      "code": "2593.0",
      "description": "Manufacture of wire products, chain and springs",
      "confidence": 0.08
    },
    {
      "code": "2591.0",
      "description": "Manufacture of steel drums and similar containers",
      "confidence": 0.03
    }
  ]
}
```

### Predict Batch of Businesses

```
POST /predict/batch
```

Request body:
```json
{
  "businesses": [
    {
      "bus_name": "Acme Widget Manufacturing",
      "description": "Manufacturing of metal widgets and gadgets for industrial use"
    },
    {
      "bus_name": "Fresh Foods Market",
      "description": "Retail sale of fresh fruits, vegetables and organic groceries"
    }
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "bus_name": "Acme Widget Manufacturing",
      "description": "Manufacturing of metal widgets and gadgets for industrial use",
      "predicted_code": "2599.0",
      "industry_description": "Manufacture of other fabricated metal products n.e.c.",
      "confidence": 0.87,
      "confidence_grade": "high",
      "alternatives": [
        {
          "code": "2593.0",
          "description": "Manufacture of wire products, chain and springs",
          "confidence": 0.08
        },
        {
          "code": "2591.0",
          "description": "Manufacture of steel drums and similar containers",
          "confidence": 0.03
        }
      ]
    },
    {
      "bus_name": "Fresh Foods Market",
      "description": "Retail sale of fresh fruits, vegetables and organic groceries",
      "predicted_code": "4721.0",
      "industry_description": "Retail sale of food in specialized stores",
      "confidence": 0.92,
      "confidence_grade": "very_high",
      "alternatives": [
        {
          "code": "4711.0",
          "description": "Retail sale in non-specialized stores with food, beverages or tobacco predominating",
          "confidence": 0.05
        },
        {
          "code": "4781.0",
          "description": "Retail sale via stalls and markets of food, beverages and tobacco products",
          "confidence": 0.02
        }
      ]
    }
  ]
}
```

### Predict from CSV File

```
POST /predict/csv
```

Upload a CSV file with columns `bus_name` and `description`. Returns a CSV file with the original data plus prediction columns.

Example cURL command:
```bash
curl -X POST "http://localhost:8000/predict/csv" \
     -H "accept: */*" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@businesses.csv;type=text/csv"
```

## Confidence Grades

The API uses the following confidence grades:

- `very_low`: confidence < 0.5
- `low`: confidence < 0.7
- `medium`: confidence < 0.8
- `high`: confidence < 0.9
- `very_high`: confidence >= 0.9