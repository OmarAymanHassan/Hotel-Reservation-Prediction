import traceback
import sys



class CustomException(Exception):
    def __init__(self, error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message , error_details)

    @staticmethod
    def get_detailed_error_message(error_message , error_details):
        # exception_info return 3 things , but we need only the last thing whihc
        # is realted to the treacback thing
        '''
        Traceback (most recent call last):
        File "D:\ITI\DS Track\Machine Learning\projects\Hotel Reservation Prediction\test_error.py", line 2, in <module>
            print(10/0)
                ~~^~
        ZeroDivisionError: division by zero
        (hotel-reservation-prediction) PS D:\ITI\DS Track\Machine Learning\projects\Hotel Reservation Prediction> 

        
        '''

        _ , _ , exception_tracback = error_details.exc_info()

        # tb: traceback
        file_name = exception_tracback.tb_frame.f_code.co_filename

        # getting the line number 
        line_number = exception_tracback.tb_lineno

        # returining the error message

        return f"Error Occured in {file_name} , Line {line_number} : {error_message}"




    def __str__(self):
        return self.error_message        
    
    # __str__ : used to give message representation 
    