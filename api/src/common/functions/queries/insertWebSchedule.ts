import pg from 'pg';
import { postgresConnection } from '../../constants/postgresConnection.js';

export const insertWebSchedule = async (
   url: string,
   html: string,
   timeRetrieved: number,
   season: number,
   isComplete: boolean,
) => {
   const { Client } = pg;
   const client = new Client(postgresConnection);
   await client.connect();
   const result = await client.query(
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
   await client.end();
   return result;
}