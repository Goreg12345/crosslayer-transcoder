from typing import Callable
import os

import torch
from torch.utils.data import IterableDataset


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    num_workers = worker_info.num_workers  # number of workers
    pointer_start = int(worker_id / num_workers * len(dataset.hf_dataset))
    dataset.set_token_pointer(pointer_start)


class TextDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset,
        to_tokens: Callable,
        batch_size,
        drop_last_batch=False,
        hf_text_accessor="text",
        seq_len=128,
    ):
        """
        Takes a huggingface dataset and returns batches of tokens
        :param hf_dataset: huggingface dataset that contains the text
        :param to_tokens: function that converts text to tokens, e.g. the tokenizer function or HookedTransformer.to_tokens()
        :param batch_size: batch size
        :param drop_last_batch: if True, the last batch will be dropped if it's smaller than batch_size
        :param hf_text_accessor: str, key to access the text in the hf_dataset
        :param seq_len: int, sequence length per sample in the batch
        returns batches of shape (batch_size, seq_len), filled with tokens
        """
        self.hf_dataset = hf_dataset
        self.to_tokens = to_tokens
        self.token_pointer = 0
        self.drop_last_batch = drop_last_batch
        self.hf_text_accessor = hf_text_accessor
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.tokens = []

        dataset_len = len(self.hf_dataset)

    def set_token_pointer(self, pointer_start):  # for multi-process dataloader
        # if e.g. 2 workers, the first worker will start at 0, the second at 1/2 of the dataset
        self.token_pointer = pointer_start

    def __iter__(self):
        return self

    def __next__(self):
        batch = torch.empty(
            (self.batch_size, self.seq_len), dtype=torch.long
        )  # .to('cuda')

        while True:
            # if dataset is exhausted, stop
            if self.token_pointer == len(self.hf_dataset):
                if self.drop_last_batch:
                    raise StopIteration
                else:
                    return batch[: self.batch_pointer]

            # get a new sample and add it to the batch
            # self.tokens += self.to_tokens(self.hf_dataset[self.token_pointer][self.hf_text_accessor], prepend_bos=False)[0]
            self.tokens += self.to_tokens(
                self.hf_dataset[self.token_pointer][self.hf_text_accessor],
               # prepend_bos=False,
            )['input_ids']
            # self.tokens += torch.randint(0, 50257, (1024,)).tolist()
            # self.hf_dataset[self.token_pointer]  # self.permutation[self.token_pointer].view(-1)]  # [self.hf_text_accessor]
            # self.permutation[self.token_pointer].view(-1)
            # self.hf_dataset[self.permutation[self.token_pointer].view(-1)]
            # test_str = 'Port-au-Prince, Haiti (CNN) -- Earthquake victims, writhing in pain and grasping at life, watched doctors and nurses walk away from a field hospital Friday night after a Belgian medical team evacuated the area, saying it was concerned about security.\n\nThe decision left CNN Chief Medical Correspondent Sanjay Gupta as the only doctor at the hospital to get the patients through the night.\n\nCNN initially reported, based on conversations with some of the doctors, that the United Nations ordered the Belgian First Aid and Support Team to evacuate. However, Belgian Chief Coordinator Geert Gijs, a doctor who was at the hospital with 60 Belgian medical personnel, said it was his decision to pull the team out for the night. Gijs said he requested U.N. security personnel to staff the hospital overnight, but was told that peacekeepers would only be able to evacuate the team.\n\nHe said it was a "tough decision" but that he accepted the U.N. offer to evacuate after a Canadian medical team, also at the hospital with Canadian security officers, left the site Friday afternoon. The Belgian team returned Saturday morning.\n\nGijs said the United Nations has agreed to provide security for Saturday night. The team has requested the Belgian government to send its own troops for the field hospital, which Gijs expects to arrive late Sunday.\n\nResponding to the CNN report that Gupta was the only doctor left at the Port-au-Prince field hospital, U.N. spokesman Martin Nesirky said Saturday that the world body\'s mission in Haiti did not order any medical team to leave. If the team left, it was at the request of their own organization, he said.\n\nEdmond Mulet, the U.N. assistant secretary general for peacekeeping operations, told reporters later that local security officers deemed the makeshift hospital unsafe.\n\n"It seems that we\'ve heard some reports in the international media that the United Nations asked or forced some medical teams to not work any more in some clinic -- that is not true, that is completely untrue," Mulet said Saturday.\n\nCNN video from the scene Friday night shows the Belgian team packing up its supplies and leaving with an escort of blue-helmeted U.N. peacekeepers in marked trucks.\n\nView or add to CNN\'s database of missing persons in Haiti\n\nGupta -- assisted by other CNN staffers, security personnel and at least one Haitian nurse who refused to leave -- assessed the needs of the 25 patients, but there was little they could do without supplies.\n\nMore people, some in critical condition, were trickling in late Friday.\n\n"I\'ve never been in a situation like this. This is quite ridiculous," Gupta said.\n\nWith a dearth of medical facilities in Haiti\'s capital, ambulances had nowhere else to take patients, some of whom had suffered severe trauma -- amputations and head injuries -- under the rubble. Others had suffered a great deal of blood loss, but there were no blood supplies left at the clinic.\n\nGupta feared that some would not survive the night.\n\nHe and the others stayed with the injured all night, after the medical team had left and after the generators gave out and the tents turned pitch black.\n\nGupta monitored patients\' vital signs, administered painkillers and continued intravenous drips. He stabilized three new patients in critical condition.\n\nAt 3:45 a.m., he posted a message on Twitter: "pulling all nighter at haiti field hosp. lots of work, but all patients stable. turned my crew into a crack med team tonight."\n\nAre you in Haiti and safe? Share your photos\n\nHe said the Belgian doctors did not want to leave their patients behind but were ordered out by the United Nations, which sent buses to transport them.\n\n"There is concern about riots not far from here -- and this is part of the problem," Gupta said.\n\nThere have been scattered reports of violence throughout the capital.\n\n"What is striking to me as a physician is that patients who just had surgery, patients who are critically ill, are essentially being left here, nobody to care for them," Gupta said.\n\nSandra Pierre, a Haitian who has been helping at the makeshift hospital, said the medical staff took most of the supplies with them.\n\n"All the doctors, all the nurses are gone," she said. "They are expected to be back tomorrow. They had no plan on leaving tonight. It was an order that came suddenly."\n\nShe told Gupta, "It\'s just you."\n\nA 7.0 magnitude earthquake flattened Haiti\'s capital city Tuesday afternoon, affecting as many as 3 million people as it fanned out across the island nation. Tens of thousands of people are feared dead.\n\nHaiti, the poorest nation in the Western hemisphere, lacked adequate medical resources even before the disaster and has been struggling this week to tend to huge numbers of injured. The clinic, set up under several tents, was a godsend to the few who were lucky to have been brought there.\n\nRetired Army Lt. Gen. Russel Honore, who led relief efforts for Hurricane Katrina in 2005, said the evacuation of the clinic\'s medical staff was unforgivable.\n\n"Search and rescue must trump security," Honoré said. "I\'ve never seen anything like this before in my life. They need to man up and get back in there."\n\nHonoré drew parallels between the tragedy in New Orleans, Louisiana, and in Port-au-Prince. But even in the chaos of Katrina, he said, he had never seen medical staff walk away.\n\n"I find this astonishing these doctors left," he said. "People are scared of the poor."\n\nCNN\'s Justine Redman, Danielle Dellorto and John Bonifield contributed to this report.'
            # self.tokens += self.to_tokens(test_str, prepend_bos=False)[0]
            self.token_pointer += 1

            # fill the batch row by row with tokens until we need to sample more or the batch is full
            while len(self.tokens) > self.seq_len:
                batch[self.batch_pointer] = torch.tensor(self.tokens[: self.seq_len])
                self.tokens = self.tokens[self.seq_len :]
                self.batch_pointer += 1
                if self.batch_pointer == self.batch_size:
                    break

            # if batch is full, return it
            if self.batch_pointer == self.batch_size:
                self.batch_pointer = 0
                return batch
