import pg from 'pg';
import { postgresConnection } from '../../constants/postgresConnection.js';

export const updateWebSchedule = async (
   webScheduleId: number,
   html: string,
   timeRetrieved: number,
   isComplete: boolean,
) => {
   const { Client } = pg;
   const client = new Client(postgresConnection);
   await client.connect();
   const result = await client.query(
      `
         UPDATE
            web_schedule
         SET
            html = $1
            ,time_retrieved = $2
            ,is_complete = $3
         WHERE
            web_schedule_id = $4
      `,
      [
         html.trim(),
         timeRetrieved,
         isComplete,
         webScheduleId,
      ],
   )
   await client.end();
   return result;
}