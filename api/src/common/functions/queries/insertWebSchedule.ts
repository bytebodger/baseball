import { dbClient } from '../../constants/dbClient.js';

export const insertWebSchedule = async (
   hasBeenPlayed: boolean,
   html: string,
   season: number,
   timeProcessed: number | null,
   timeRetrieved: number,
   url: string,
) => {
   return await dbClient.query(
      `
         INSERT INTO
           web_schedule
            (
               has_been_played
               ,html
               ,season
               ,time_processed 
               ,time_retrieved
               ,url
            )
         VALUES 
            ($1, $2, $3, $4, $5, $6)
      `,
      [
         hasBeenPlayed,
         html.trim(),
         season,
         timeProcessed,
         timeRetrieved,
         url.trim(),
      ],
   )
}