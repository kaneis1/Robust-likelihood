"""Hand-authored alarm drafts for the seeded train sample.

Time, negation, and the requested action stay intact on meaning-preserving versions.
Meaning-change controls flip the action and store the new label.
polite_restatement is not authored.
"""

AUTHORED_IDS = [
    "722",
    "1721",
    "3180",
    "3498",
    "4929",
    "650",
    "1037",
    "4529",
    "5120",
    "6019",
    "1002",
    "2878",
    "3537",
    "3828",
    "4970",
]

DRAFTS = {
    "722": {
        "intent": "alarm_set",
        "utt": "make an alarm to wake me up after five hours",
        "paraphrase": "create an alarm to wake me up after five hours",
        "minor_typo": "make an alram to wake me up after five hours",
        "meaning_change": {
            "utt": "cancel an alarm to wake me up after five hours",
            "intent": "alarm_remove",
        },
        "preserve": ["after five hours"],
    },
    "1721": {
        "intent": "alarm_set",
        "utt": "set alarm of whole week morning ten am in april month",
        "paraphrase": "set an alarm for the whole week at ten am in the morning in april",
        "minor_typo": "set alarm of whole week morning ten am in apirl month",
        "meaning_change": {
            "utt": "cancel alarm of whole week morning ten am in april month",
            "intent": "alarm_remove",
        },
        "preserve": ["ten am"],
    },
    "3180": {
        "intent": "alarm_set",
        "utt": "i'd like an alarm set for this saturday at ten",
        "paraphrase": "i would like an alarm set for this saturday at ten",
        "minor_typo": "i'd like an alram set for this saturday at ten",
        "meaning_change": {
            "utt": "i'd like an alarm cancelled for this saturday at ten",
            "intent": "alarm_remove",
        },
        "preserve": ["saturday at ten"],
    },
    "3498": {
        "intent": "alarm_set",
        "utt": "i need to get up at seven am",
        "paraphrase": "i have to get up at seven am",
        "minor_typo": "i ned to get up at seven am",
        "meaning_change": {
            "utt": "cancel the alarm to get up at seven am",
            "intent": "alarm_remove",
        },
        "preserve": ["seven am"],
    },
    "4929": {
        "intent": "alarm_set",
        "utt": "turn on my first scheduled alarm",
        "paraphrase": "enable my first scheduled alarm",
        "minor_typo": "turn on my first schedled alarm",
        "meaning_change": {
            "utt": "turn off my first scheduled alarm",
            "intent": "alarm_remove",
        },
        "preserve": ["first"],
    },
    "650": {
        "intent": "alarm_query",
        "utt": "i need to set an alarm how many do i have set",
        "paraphrase": "i need to set an alarm, how many do i have set",
        "minor_typo": "i need to set an alram how many do i have set",
        "meaning_change": {
            "utt": "delete the alarms i have set",
            "intent": "alarm_remove",
        },
        "preserve": ["how many do i have set"],
    },
    "1037": {
        "intent": "alarm_query",
        "utt": "what alarms do i have set for thursday",
        "paraphrase": "which alarms do i have set for thursday",
        "minor_typo": "what alrams do i have set for thursday",
        "meaning_change": {
            "utt": "delete the alarms i have set for thursday",
            "intent": "alarm_remove",
        },
        "preserve": ["thursday"],
    },
    "4529": {
        "intent": "alarm_query",
        "utt": "what's my next scheduled alarm",
        "paraphrase": "what is my next scheduled alarm",
        "minor_typo": "what's my next scheduld alarm",
        "meaning_change": {
            "utt": "cancel my next scheduled alarm",
            "intent": "alarm_remove",
        },
        "preserve": ["next"],
    },
    "5120": {
        "intent": "alarm_query",
        "utt": "check if any alarm is there after five am",
        "paraphrase": "check whether any alarm is there after five am",
        "minor_typo": "check if any alram is there after five am",
        "meaning_change": {
            "utt": "set an alarm for after five am",
            "intent": "alarm_set",
        },
        "preserve": ["after five am"],
    },
    "6019": {
        "intent": "alarm_query",
        "utt": "would like to know the alarm you sent",
        "paraphrase": "i would like to know the alarm you sent",
        "minor_typo": "would like to knwo the alarm you sent",
        "meaning_change": {
            "utt": "would like to cancel the alarm you sent",
            "intent": "alarm_remove",
        },
        "preserve": ["the alarm you sent"],
    },
    "1002": {
        "intent": "alarm_remove",
        "utt": "delete my alarm for eight am",
        "paraphrase": "remove my alarm for eight am",
        "minor_typo": "delete my alram for eight am",
        "meaning_change": {
            "utt": "set my alarm for eight am",
            "intent": "alarm_set",
        },
        "preserve": ["eight am"],
    },
    "2878": {
        "intent": "alarm_remove",
        "utt": "please cancel all alarms for tomorrow",
        "paraphrase": "please delete all alarms for tomorrow",
        "minor_typo": "please cancel all alrams for tomorrow",
        "meaning_change": {
            "utt": "please set all alarms for tomorrow",
            "intent": "alarm_set",
        },
        "preserve": ["tomorrow"],
    },
    "3537": {
        "intent": "alarm_remove",
        "utt": "turn off my alarms",
        "paraphrase": "switch off my alarms",
        "minor_typo": "tun off my alarms",
        "meaning_change": {
            "utt": "turn on my alarms",
            "intent": "alarm_set",
        },
        "preserve": ["my alarms"],
    },
    "3828": {
        "intent": "alarm_remove",
        "utt": "i need to delete the alarm for church on sundays at eleven am",
        "paraphrase": "i need to remove the alarm for church on sundays at eleven am",
        "minor_typo": "i need to delete the alarm for churhc on sundays at eleven am",
        "meaning_change": {
            "utt": "i need to set the alarm for church on sundays at eleven am",
            "intent": "alarm_set",
        },
        "preserve": ["sundays at eleven am"],
    },
    "4970": {
        "intent": "alarm_remove",
        "utt": "remove my work alarm",
        "paraphrase": "delete my work alarm",
        "minor_typo": "remov my work alarm",
        "meaning_change": {
            "utt": "set my work alarm",
            "intent": "alarm_set",
        },
        "preserve": ["work alarm"],
    },
}
