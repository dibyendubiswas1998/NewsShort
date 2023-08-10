from predictions import prediction

input_text = """
    The opposition moved a no-confidence motion against the Modi government on July 26 which was taken up by Lok Sabha Speaker Om Birla. Prime Minister Narendra Modi will mark his presence in the Lok Sabha on August 10 to reply to the no-confidence motion moved by the Opposition against the NDA government, Defence Minister Rajnath Singh said on Wednesday. 
    The PM will be present in the House tomorrow to reply to the no-confidence motion,” the Union minister told the Lower House.
    Just before the adjournment of the House, the Union Minister confirmed the same.
    The opposition moved a no-confidence motion against the Modi government on July 26 which was taken up by Lok Sabha Speaker Om Birla.
    However, Modi's government won't lose the vote as his Bharatiya Janata Party (BJP) and its allies have a majority in the Lok Sabha.
    Any Lok Sabha MP, who has the support of 50 colleagues, can, at any point of time, introduce a motion of no-confidence against the Council of Ministers.
    Thereafter, a discussion on the motion takes place. MPs who support the motion highlight the government’s shortcomings, and the Treasury Benches respond to the issues they raise. Ultimately, voting takes place and if the motion is successful, the government is forced to vacate the office.
    Notably, the NDA has a commendable majority with a number of 331 MPs out of which the BJP has 303 MPs while the combined strength of the Opposition bloc I.N.D.I.A is 144. The numbers of unaligned parties’ MPs are 70 in the Lower House.
    This is the second time Prime Minister Narendra Modi is facing a no-confidence motion.
    The first such motion against the Modi government was introduced in 2018 over granting a special category status to Andhra Pradesh which was later defeated.
"""

summary = prediction(text=input_text, max_length=250)
print(summary)

