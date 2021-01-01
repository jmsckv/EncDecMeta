
def fancy_print(message, asterix_line_repetitions=1, start_with_empty_line=True, end_with_empty_line=True, return_string=False):
    message_augmented = ''
    if start_with_empty_line:
        message_augmented += '\n'
    for i in range(asterix_line_repetitions):
        message_augmented += "*" * len(message)
    message_augmented += '\n' + message + '\n'
    for i in range(asterix_line_repetitions):
        message_augmented += "*" * len(message)
    if end_with_empty_line:
        message_augmented += '\n'
    if return_string:
        return message_augmented
    else:
        print(message_augmented)