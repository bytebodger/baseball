export interface WebSchedule {
   has_been_played: boolean,
   html: string,
   season: number,
   time_processed: number | null,
   time_retrieved: number,
   url: string,
   web_schedule_id: number,
}