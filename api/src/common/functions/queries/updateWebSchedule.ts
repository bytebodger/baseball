import { dbClient } from '../../constants/dbClient.js';

export const updateWebSchedule = async (
   webScheduleId: number,
   html: string,
   timeRetrieved: number,
   isComplete: boolean,
) => {
   return await dbClient.query(
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
}