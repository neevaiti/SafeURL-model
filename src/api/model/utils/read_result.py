def read_result(prediction):
    """
    Function to read the prediction and return the corresponding response.
    Parameters:
    - prediction: the predicted value
    Returns:
    - response: the corresponding response based on the prediction value
    """
    if int(prediction[0]) == 0:
        return 0  # Safe
    elif int(prediction[0]) == 1:
        return 1  # Phishing