def read_result(prediction):
    """
    Function to read the prediction and return the corresponding response.
    Parameters:
    - prediction: the predicted value (0 or 1)
    Returns:
    - response: 'safe' for 0, 'phishing' for 1
    """
    if int(prediction) == 0:
        return 'safe'
    elif int(prediction) == 1:
        return 'phishing'
    else:
        raise ValueError("Invalid prediction value")
