from pydantic import BaseModel

class CustomerData(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: int

class PredictionResponse(BaseModel):
    transaction_id: str
    customer_id: str
    default_probability: float
    risk_tier: str
    action: str