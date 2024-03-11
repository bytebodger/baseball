import { dbClient } from '../../constants/dbClient.js';

export const insertWebSchedule = async (
   url: string,
   html: string,
   timeRetrieved: number,
   season: number,
   isComplete: boolean,
) => {
   return await dbClient.query(
      `
         INSERT INTO
           web_schedule
            (
               url
               ,html
               ,time_retrieved
               ,season
               ,is_complete
            )
         VALUES 
            ($1, $2, $3, $4, $5)
      `,
      [
         url.trim(),
         html.trim(),
         timeRetrieved,
         season,
         isComplete,
      ],
   )
}