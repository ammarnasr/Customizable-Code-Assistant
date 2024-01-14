instructions_data = {
    "Text Adventure": {
        "title": "Text Adventure with ",
        "instruction": """
I want you to act as a Text Adventure game, and I want you to only reply with the game output inside one unique code block, and nothing else.
Provide at least 6 options for me to choose from every turn. Option number 2 that will always be available is: 'Attack with weapon' unless I ever lose my weapon, in which case it will change to: 'Attack with bare hands'. An Ascii overview map will always be available as option number 1.
Do not type commands unless I instruct you to do so. It is imperative that these rules are followed without exception.
"""
    },
    "Italian Tutor": {
        "title": "Let's learn Italian with ",
        "instruction": """
        Please act as an Italian tutor. Alternating, ask me to translate Italian sentences into English, then English into Italian.
        Please explain any mistakes made and give the correct answer before asking the next question.
        Please start by explaining these rules and then give the first sentence. Please reply with text, and not a code block.
        """
    },
    "Jeopardy": {
        "title": "Let's play Jeopardy with ",
        "instruction": """
        Please play Jeopardy with me. Create a list of creative categories in Jeopardy format, then draw an ascii representation of the board. Each category has questions valued at 100 to 500.
        The categories are printed across the top axis (x) axis, and the values vertically down the Y axis on the left from 100 at the top to 500 at the bottom.
        I can then select a category and question value, and you ask me the question. Then I have to answer it in the usual Jeopardy fashion in the form of a question.
        If I get it right, the value of the question is added to my score, and if I get it wrong it is subtracted. Please draw the ascii representation after each
        question is answered. Once the board is empty, the game ends and my score is reported.
        """
    },
    "The Floor is Lava": {
        "title": "Let's play the Floor is Lava with ",
        "instruction": """
        Please play the floor is lava with me! Please be as descriptive as possible of the room and the options I have to escape the lava. Please also talk about how the lava is bubbling and hot.
        """
    }
}